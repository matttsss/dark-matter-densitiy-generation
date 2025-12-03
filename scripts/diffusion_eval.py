"""
Full evaluation of diffusion model trained in AstroPT embedding space.

What this script does:
1. Loads:
   - Finetuned AstroPT model (for embeddings)
   - Trained DDPM model (for sampling embeddings)

2. Builds:
   - Real embeddings + (mass, label) from BAHAMAS
   - Fake embeddings sampled from DDPM, conditioned on (mass, label) pairs

3. Evaluates:
   A. UMAP projections:
      - real vs fake
      - colored by mass
      - colored by label

   B. FID-style distance between real and fake embeddings

   C. Linear probe tests on mass:
      - Train on real, test on real (baseline)
      - Train on real, test on fake (does fake encode mass?)
      - Train on real+fake, test on held-out real (augmentation benefit)
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SKLLinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from umap import UMAP

from generative_model.DDPM import DDPM
from generative_model.DiT import DiT
from model_utils import load_model
from embedings_utils import merge_datasets, compute_embeddings


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_astropt_encoder(ckpt_path: str, device: torch.device):
    """
    Load finetuned AstroPT model that exposes `get_embeddings`.
    """
    print(f"Loading finetuned AstroPT encoder from: {ckpt_path}")
    try:
        model = load_model(ckpt_path, device=device)
    except TypeError:
        # Fallback for signature load_model(path, device, lora_rank, output_dim, strict)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", ckpt.get("model_args", {}))
        lora_rank = config.get("lora_rank", 0)
        output_dim = config.get("output_dim", 0)
        print(f"  Detected lora_rank={lora_rank}, output_dim={output_dim} from checkpoint metadata")
        model = load_model(ckpt_path, device, lora_rank, output_dim)

    model.eval()
    return model


def get_real_embeddings(
    astropt_model,
    device: torch.device,
    nb_points: int,
    batch_size: int,
    labels_name=None,
):
    """
    Compute real embeddings and corresponding labels from BAHAMAS.
    """
    if labels_name is None:
        labels_name = ["mass", "label"]

    print("Loading BAHAMAS dataset for REAL embeddings...")
    dataset = (
        merge_datasets([
            "data/DarkData/BAHAMAS/bahamas_0.1.pkl",
            "data/DarkData/BAHAMAS/bahamas_0.3.pkl",
            "data/DarkData/BAHAMAS/bahamas_1.pkl",
            "data/DarkData/BAHAMAS/bahamas_cdm.pkl",
        ])
        .select_columns(["images", "images_positions", *labels_name])
        .shuffle(seed=42)
    )

    if nb_points is not None:
        nb_points = min(nb_points, len(dataset))
        dataset = dataset.select(range(nb_points))
        print(f"Using {nb_points} real samples (out of {len(dataset)} available)")
    else:
        print(f"Using all {len(dataset)} real samples")

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    print("Computing REAL embeddings with AstroPT encoder...")
    embeddings, labels = compute_embeddings(
        astropt_model,
        dl,
        device,
        labels_name,
        disable_tqdm=True,
    )
    embeddings = embeddings.cpu()
    labels = {k: v.cpu() for k, v in labels.items()}

    print(f"  -> Real embeddings shape: {embeddings.shape}")
    return embeddings, labels


def load_diffusion_model(diff_ckpt_path: str, device: torch.device):
    """
    Instantiate DDPM with correct DiT configuration by reading the checkpoint.
    We infer:
      - num_patches, hidden_size from eps_model.pos_embed
      - patch_size from eps_model.x_embedder.weight
      - input_size = num_patches * patch_size

    This guarantees compatibility with the trained checkpoint.
    """
    print(f"Loading diffusion model from: {diff_ckpt_path}")
    state_dict = torch.load(diff_ckpt_path, map_location=device)

    # Infer num_patches & hidden_size from positional embedding
    pos_embed = state_dict["eps_model.pos_embed"]  # [1, num_patches, hidden_size]
    num_patches = pos_embed.shape[1]
    hidden_size = pos_embed.shape[2]

    # Infer patch_size from x_embedder
    x_embed_weight = state_dict["eps_model.x_embedder.weight"]  # [hidden_size, patch_size]
    patch_size = x_embed_weight.shape[1]

    # True input size used at training time
    input_size = num_patches * patch_size

    print(f"  Detected from checkpoint:")
    print(f"    num_patches = {num_patches}")
    print(f"    hidden_size = {hidden_size}")
    print(f"    patch_size  = {patch_size}")
    print(f"    input_size  = {input_size}  (data_dim for DDPM)")

    # Build DDPM with correct data_dim
    diffusion_model = DDPM(data_dim=(input_size,))

    # Override eps_model with matching DiT config
    diffusion_model.eps_model = DiT(
        input_size=input_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        # depth and num_heads are inferred from checkpoint structure,
        # so defaults must match what was used at training time.
    )

    diffusion_model.load_state_dict(state_dict)
    diffusion_model.to(device)
    diffusion_model.eval()
    return diffusion_model, input_size


@torch.no_grad()
def sample_diffusion_embeddings(
    diffusion_model,
    device: torch.device,
    cond_mass: torch.Tensor,
    cond_label: torch.Tensor,
    batch_size: int,
):
    """
    Sample embeddings from diffusion model conditioned on mass + label.

    cond_mass: [N]
    cond_label: [N]
    Returns: [N, data_dim] tensor (on CPU)
    """
    assert cond_mass.shape == cond_label.shape
    N = cond_mass.shape[0]

    all_samples = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        m = cond_mass[start:end].to(device)
        l = cond_label[start:end].to(device)
        cond = torch.stack([m, l], dim=1)  # [B, 2]
        samples = diffusion_model.sample(cond)  # [B, data_dim]
        all_samples.append(samples.cpu())

    fake_embeddings = torch.cat(all_samples, dim=0)
    print(f"  -> Fake embeddings shape: {fake_embeddings.shape}")
    return fake_embeddings


def compute_fid_torch(real: torch.Tensor, fake: torch.Tensor) -> float:
    """
    Compute a FID-like distance between real and fake embeddings.
    Implementation uses eigen-decomposition; no SciPy needed.
    real, fake: [N, D] tensors on CPU.
    """
    real = real.float()
    fake = fake.float()

    mu_r = real.mean(dim=0)
    mu_f = fake.mean(dim=0)

    # Centered
    r_centered = real - mu_r
    f_centered = fake - mu_f

    # Covariance: [D, D]
    sigma_r = r_centered.t().mm(r_centered) / (real.shape[0] - 1)
    sigma_f = f_centered.t().mm(f_centered) / (fake.shape[0] - 1)

    # Product (may not be exactly symmetric)
    cov_prod = sigma_r.mm(sigma_f)
    cov_prod = (cov_prod + cov_prod.t()) / 2  # symmetrize

    # Matrix square root via eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov_prod)
    eigvals = torch.clamp(eigvals, min=0.0)
    sqrt_cov_prod = eigvecs.mm(torch.diag(torch.sqrt(eigvals))).mm(eigvecs.t())

    diff = mu_r - mu_f
    fid = diff.dot(diff) + torch.trace(sigma_r + sigma_f - 2 * sqrt_cov_prod)
    return float(fid.item())


def run_linear_probe_eval(
    real_embeddings: torch.Tensor,
    real_labels: dict,
    fake_embeddings: torch.Tensor,
    fake_mass: torch.Tensor,
):
    """
    Linear probe evaluation on MASS.

    Returns dict of metrics + aux info for plotting.
    """
    real_emb_np = real_embeddings.numpy()
    real_mass_np = real_labels["mass"].numpy().reshape(-1)

    fake_emb_np = fake_embeddings.numpy()
    fake_mass_np = fake_mass.numpy().reshape(-1)

    N = real_emb_np.shape[0]
    halfway = N // 2

    X_train = real_emb_np[:halfway]
    y_train = real_mass_np[:halfway]
    X_test_real = real_emb_np[halfway:]
    y_test_real = real_mass_np[halfway:]

    # 1) Baseline: train on REAL, test on REAL
    probe_base = SKLLinearRegression().fit(X_train, y_train)
    y_pred_real = probe_base.predict(X_test_real)
    mse_real = mean_squared_error(y_test_real, y_pred_real)
    r2_real = r2_score(y_test_real, y_pred_real)

    # 2) Train on REAL, test on FAKE (target = conditional mass)
    y_pred_fake = probe_base.predict(fake_emb_np)
    mse_fake = mean_squared_error(fake_mass_np, y_pred_fake)
    r2_fake = r2_score(fake_mass_np, y_pred_fake)

    # 3) Train on REAL+FAKE, test on held-out REAL
    X_train_aug = np.concatenate([X_train, fake_emb_np], axis=0)
    y_train_aug = np.concatenate([y_train, fake_mass_np], axis=0)

    probe_aug = SKLLinearRegression().fit(X_train_aug, y_train_aug)
    y_pred_real_aug = probe_aug.predict(X_test_real)
    mse_real_aug = mean_squared_error(y_test_real, y_pred_real_aug)
    r2_real_aug = r2_score(y_test_real, y_pred_real_aug)

    metrics = {
        "mse_real_baseline": mse_real,
        "r2_real_baseline": r2_real,
        "mse_fake_under_real_probe": mse_fake,
        "r2_fake_under_real_probe": r2_fake,
        "mse_real_with_aug": mse_real_aug,
        "r2_real_with_aug": r2_real_aug,
    }

    return metrics, {
        "probe_base": probe_base,
        "probe_aug": probe_aug,
        "X_test_real": X_test_real,
        "y_test_real": y_test_real,
        "y_pred_real": y_pred_real,
        "y_pred_real_aug": y_pred_real_aug,
    }


def plot_umap_and_probes(
    real_embeddings,
    real_labels,
    fake_embeddings,
    fake_mass,
    fake_label,
    probe_info,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- UMAP ----------------------
    print("Computing UMAP projections for real vs fake...")
    real_np = real_embeddings.numpy()
    fake_np = fake_embeddings.numpy()

    X = np.concatenate([real_np, fake_np], axis=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = umap.fit_transform(X_scaled)

    real_coords = coords[:real_np.shape[0]]
    fake_coords = coords[real_np.shape[0]:]

    real_mass = real_labels["mass"].numpy()
    real_label = real_labels["label"].numpy()

    # 1) Domain (real vs fake)
    plt.figure(figsize=(6, 5))
    plt.scatter(real_coords[:, 0], real_coords[:, 1], s=4, alpha=0.5, label="Real")
    plt.scatter(fake_coords[:, 0], fake_coords[:, 1], s=4, alpha=0.5, label="Fake")
    plt.legend()
    plt.title("UMAP: Real vs Fake Embeddings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_real_vs_fake.png"), dpi=200)
    plt.close()

    # 2) Real only, colored by mass
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(real_coords[:, 0], real_coords[:, 1], c=real_mass, s=4, alpha=0.7, cmap="viridis")
    plt.colorbar(sc, label="Mass")
    plt.title("UMAP (REAL) colored by mass")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_real_mass.png"), dpi=200)
    plt.close()

    # 3) Real only, colored by label
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(real_coords[:, 0], real_coords[:, 1], c=real_label, s=4, alpha=0.7, cmap="tab10")
    plt.colorbar(sc, label="Label (σ/m)")
    plt.title("UMAP (REAL) colored by label")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_real_label.png"), dpi=200)
    plt.close()

    # ---------------------- Probe scatter ----------------------
    X_test_real = probe_info["X_test_real"]
    y_test_real = probe_info["y_test_real"]
    y_pred_real = probe_info["y_pred_real"]
    y_pred_real_aug = probe_info["y_pred_real_aug"]

    # Baseline probe: real vs predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_real, y_pred_real, s=8, alpha=0.3)
    lims = [
        min(y_test_real.min(), y_pred_real.min()),
        max(y_test_real.max(), y_pred_real.max()),
    ]
    plt.plot(lims, lims, "r--", label="y = x")
    plt.xlabel("True mass (held-out real)")
    plt.ylabel("Pred mass (probe trained on real)")
    plt.title("Baseline linear probe on REAL embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probe_real_vs_real.png"), dpi=200)
    plt.close()

    # Augmented probe: real vs predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_real, y_pred_real_aug, s=8, alpha=0.3)
    lims = [
        min(y_test_real.min(), y_pred_real_aug.min()),
        max(y_test_real.max(), y_pred_real_aug.max()),
    ]
    plt.plot(lims, lims, "r--", label="y = x")
    plt.xlabel("True mass (held-out real)")
    plt.ylabel("Pred mass (probe trained on real+fake)")
    plt.title("Linear probe with REAL+FAKE augmentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probe_real_augmented.png"), dpi=200)
    plt.close()

    print(f"Saved UMAP & probe plots to: {output_dir}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation of diffusion model in AstroPT embedding space"
    )

    parser.add_argument(
        "--astropt_ckpt",
        type=str,
        required=True,
        help="Path to finetuned AstroPT checkpoint used for embeddings",
    )
    parser.add_argument(
        "--diffusion_ckpt",
        type=str,
        default="best_diffusion_model.pt",
        help="Path to trained diffusion model state_dict",
    )
    parser.add_argument(
        "--nb_real",
        type=int,
        default=8000,
        help="Number of REAL examples to use for evaluation",
    )
    parser.add_argument(
        "--nb_fake",
        type=int,
        default=8000,
        help="Number of FAKE samples to generate (will be limited by nb_real)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding computation and diffusion sampling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion_eval_outputs",
        help="Directory to save plots and metrics",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Evaluating on device: {device}")

    # ---------------------------------------------------------
    # 1. Load encoder + diffusion
    # ---------------------------------------------------------
    astropt_model = load_astropt_encoder(args.astropt_ckpt, device)
    diffusion_model, diff_data_dim = load_diffusion_model(args.diffusion_ckpt, device)

    # ---------------------------------------------------------
    # 2. REAL embeddings + labels
    # ---------------------------------------------------------
    real_embeddings, real_labels = get_real_embeddings(
        astropt_model,
        device,
        nb_points=args.nb_real,
        batch_size=args.batch_size,
        labels_name=["mass", "label"],
    )

    # Sanity check: embedding dimension must match diffusion data dim
    D_real = real_embeddings.shape[1]
    print(f"Real embedding dim = {D_real}, diffusion data_dim = {diff_data_dim}")
    if D_real != diff_data_dim:
        print(
            f"WARNING: Embedding dim ({D_real}) != diffusion data_dim ({diff_data_dim}).\n"
            "This suggests a mismatch between how embeddings were constructed during diffusion training\n"
            "and how they are computed here."
        )

    # ---------------------------------------------------------
    # 3. FAKE embeddings conditioned on (mass, label)
    # ---------------------------------------------------------
    cond_mass_all = real_labels["mass"]
    cond_label_all = real_labels["label"]

    N_real = cond_mass_all.shape[0]
    N_fake = min(args.nb_fake, N_real)

    perm = torch.randperm(N_real)
    idx = perm[:N_fake]

    cond_mass = cond_mass_all[idx]
    cond_label = cond_label_all[idx]

    print(f"Sampling {N_fake} FAKE embeddings conditioned on real (mass, label)...")
    fake_embeddings = sample_diffusion_embeddings(
        diffusion_model,
        device,
        cond_mass,
        cond_label,
        batch_size=args.batch_size,
    )

    # ---------------------------------------------------------
    # 4. FID-style distance
    # ---------------------------------------------------------
    print("Computing FID-style distance in embedding space...")
    fid_value = compute_fid_torch(real_embeddings, fake_embeddings)
    print(f"  FID-like distance (real vs fake embeddings): {fid_value:.4f}")

    # ---------------------------------------------------------
    # 5. Linear probe evaluation (mass)
    # ---------------------------------------------------------
    print("Running linear probe evaluation on MASS...")
    probe_metrics, probe_info = run_linear_probe_eval(
        real_embeddings,
        real_labels,
        fake_embeddings,
        cond_mass,
    )

    print("\n==== LINEAR PROBE METRICS (mass) ====")
    for k, v in probe_metrics.items():
        print(f"  {k:30s}: {v:.6f}")

    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"FID_like: {fid_value:.6f}\n")
        for k, v in probe_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"\nSaved metrics to {metrics_path}")

    # ---------------------------------------------------------
    # 6. UMAP + probe plots
    # ---------------------------------------------------------
    print("Generating UMAP & probe plots...")
    plot_umap_and_probes(
        real_embeddings,
        real_labels,
        fake_embeddings,
        cond_mass,
        cond_label,
        probe_info,
        args.output_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()