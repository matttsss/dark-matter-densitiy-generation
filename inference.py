import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

from scripts.model_utils import load_fm_model, load_astropt_model
from scripts.embedings_utils import merge_datasets
from generative_model.DDPM import DDPM


LABEL_MAP = {0.01: 0, 0.1: 1, 0.3: 2, 1.0: 3}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# map label in argument to categorical label used in model
def map_label(label: float) -> int:
    """Map physical label (0.01, 0.1, 0.3, 1) to model label (0, 1, 2, 3)."""
    # Find closest key if not exact match
    closest = min(LABEL_MAP.keys(), key=lambda x: abs(x - label))
    if abs(closest - label) > 0.01:
        print(f"Warning: label {label} mapped to closest value {closest}")
    return LABEL_MAP[closest]

DATASETS = [
    "data/BAHAMAS/bahamas_0.1.pkl",
    "data/BAHAMAS/bahamas_0.3.pkl",
    "data/BAHAMAS/bahamas_1.pkl",
    "data/BAHAMAS/bahamas_cdm.pkl",
]


def load_ddpm(checkpoint_path: str, device: torch.device) -> DDPM:
    """Load DDPM model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DDPM checkpoint not found: {checkpoint_path}")
    
    model = DDPM(
        patch_size=4,
        schedule="cosine",
        timesteps=1000,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print(f"Loaded EMA weights from epoch {checkpoint.get('epoch', '?')}")
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state dict directly")
    
    model.eval()
    return model


def generate_images(
    ddpm_model: DDPM,
    embeddings: torch.Tensor,
    use_ddim: bool = True,
    ddim_steps: int = 50,
    guidance_scale: float = 2.0,
    eta: float = 0.0,
) -> torch.Tensor:
    """Generate images from embeddings."""
    device = next(ddpm_model.parameters()).device
    embeddings = embeddings.to(device)
    
    print(f"Generating {embeddings.shape[0]} image(s)...")
    print(f"  Method: {'DDIM' if use_ddim else 'DDPM'}, steps: {ddim_steps}, guidance: {guidance_scale}")
    
    ddpm_model.eval()
    with torch.no_grad():
        if use_ddim:
            images = ddpm_model.ddim_sample(
                embeddings, steps=ddim_steps, eta=eta, guidance_scale=guidance_scale
            )
        else:
            images = ddpm_model.sample(embeddings, guidance_scale=guidance_scale)
    
    images = (images - images.min()) / (images.max() - images.min() + 1e-8)
    print(f"  Output shape: {images.shape}")
    return images


# flow model pipeline
def run_fm_pipeline(
    mass: float,
    label: float,  # can be 0.01, 0.1, 0.3, or 1.0
    fm_path: str,
    ddpm_path: str,
    device: torch.device,
    ddim_steps: int = 50,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    model_label = map_label(label)
    print(f"\n{'='*60}\nFM PIPELINE: mass={mass}, label={label} (model: {model_label})\n{'='*60}")
    
    fm_model = load_fm_model(fm_path, device)
    fm_model.eval()
    
    with torch.no_grad():
        cond = torch.tensor([[mass, model_label]], dtype=torch.float32, device=device)  # Use model_label
        embeddings = fm_model.sample_flow(cond)
    print(f"Embedding: shape={embeddings.shape}, mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
    
    ddpm_model = load_ddpm(ddpm_path, device)
    return generate_images(ddpm_model, embeddings, ddim_steps=ddim_steps, guidance_scale=guidance_scale)


def run_fm_batch(
    masses: list,
    labels: list,  # can be 0.01, 0.1, 0.3, or 1.0
    fm_path: str,
    ddpm_path: str,
    device: torch.device,
    ddim_steps: int = 50,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    model_labels = [map_label(l) for l in labels]    
    fm_model = load_fm_model(fm_path, device)
    fm_model.eval()
    ddpm_model = load_ddpm(ddpm_path, device)
    
    with torch.no_grad():
        cond = torch.tensor(list(zip(masses, model_labels)), dtype=torch.float32, device=device)  # Use model_labels
        embeddings = fm_model.sample_flow(cond)
    
    return generate_images(ddpm_model, embeddings, ddim_steps=ddim_steps, guidance_scale=guidance_scale)


# astropt pipeline

def run_astropt_pipeline(
    sample_idx: int,
    astropt_path: str,
    ddpm_path: str,
    device: torch.device,
    ddim_steps: int = 50,
    guidance_scale: float = 2.0,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """AstroPT pipeline: image → embedding → reconstruction"""
    print(f"\n{'='*60}\nASTROPT PIPELINE: sample_idx={sample_idx}\n{'='*60}")
    
    dataset = merge_datasets(DATASETS).select_columns(["images", "images_positions", "mass", "label"])
    dataset_images = merge_datasets(DATASETS, image_only=True)
    
    sample = dataset[sample_idx]
    original = np.array(dataset_images[sample_idx]["image"])
    metadata = {"mass": sample["mass"], "label": sample["label"], "idx": sample_idx}
    print(f"Sample: mass={metadata['mass']:.4f}, label={metadata['label']}")
    
    astropt_model = load_astropt_model(checkpoint_path=astropt_path, device=device)
    astropt_model.eval()
    
    image = torch.as_tensor(sample["images"]).unsqueeze(0) if torch.as_tensor(sample["images"]).dim() == 2 else torch.as_tensor(sample["images"])
    positions = torch.as_tensor(sample["images_positions"]).unsqueeze(0) if torch.as_tensor(sample["images_positions"]).dim() == 1 else torch.as_tensor(sample["images_positions"])
    
    with torch.no_grad():
        batch = {"images": image.to(device), "images_positions": positions.to(device)}
        embeddings = astropt_model.generate_embeddings(batch)["images"]
    print(f"Embedding: shape={embeddings.shape}, mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
    
    ddpm_model = load_ddpm(ddpm_path, device)
    generated = generate_images(ddpm_model, embeddings, ddim_steps=ddim_steps, guidance_scale=guidance_scale)
    
    return generated, original, metadata


# plots for flow model results
def plot_fm_result(image: np.ndarray, mass: float, label: float, save_path: str = None):
    """Plot FM generation."""
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap='viridis')
    plt.title(f"FM Generated\nmass={mass:.2f}, label={label:.0f}")
    plt.colorbar()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()




def plot_astropt_result(original: np.ndarray, generated: np.ndarray, metadata: dict, save_path: str = None):
    """Plot AstroPT original vs reconstruction."""
    from scipy.stats import pearsonr
    from skimage.transform import resize
    
    # Handle original format
    if original.ndim == 3:
        original = original[0] if original.shape[0] in [1, 3] else original[:, :, 0]
    
    orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
    if orig_norm.shape != generated.shape:
        orig_resized = resize(orig_norm, generated.shape, anti_aliasing=True)
    else:
        orig_resized = orig_norm
    
    corr, _ = pearsonr(orig_resized.flatten(), generated.flatten())
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    vmin, vmax = min(orig_norm.min(), generated.min()), max(orig_norm.max(), generated.max())
    
    im0 = axes[0].imshow(orig_norm, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original (idx={metadata['idx']})\nmass={metadata['mass']:.2f}, label={metadata['label']:.0f}")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    im1 = axes[1].imshow(generated, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Reconstruction\nCorr: {corr:.3f}")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    diff = orig_resized - generated
    im2 = axes[2].imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[2].set_title(f"Difference\nMAE: {np.abs(diff).mean():.3f}")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM inference for FM or AstroPT pipelines")
    
    parser.add_argument("--mode", type=str, choices=["fm", "astropt"], required=True)
    
    # FM mode
    parser.add_argument("--mass", type=float, help="Mass value (FM mode)")
    parser.add_argument("--label", type=float, help="Label value (FM mode)")
    parser.add_argument("--fm_path", type=str, default="model_weights/flowmodel.pt")
    parser.add_argument("--mass_min", type=float, default=13.5)
    parser.add_argument("--mass_max", type=float, default=15.5)
    
    # AstroPT mode
    parser.add_argument("--sample_idx", type=int, help="Sample index (AstroPT mode)")
    parser.add_argument("--astropt_path", type=str, default="model_weights/finetunedastroptcheckpoint.pt")
    
    # Shared
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="generated.pt")
    parser.add_argument("--save_png", type=str, default="generated.png")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ddpm_fm = "model_weights/flowmodelddpm.pt"
    ddpm_astropt = "model_weights/astroptddpm.pt"
    
    if args.mode == "fm":
        if args.label is None:
            parser.error("FM mode requires --label")
            if args.mass is None:
                parser.error("FM mode requires --mass or --mass_sweep")
            images = run_fm_pipeline(args.mass, args.label, args.fm_path, ddpm_fm, device,
                                     args.ddim_steps, args.guidance_scale)
            torch.save(images, args.output)
            plot_fm_result(images[0, 0].cpu().numpy(), args.mass, args.label, args.save_png)
    
    elif args.mode == "astropt":
        if args.sample_idx is None:
            parser.error("AstroPT mode requires --sample_idx")
        
        generated, original, metadata = run_astropt_pipeline(
            args.sample_idx, args.astropt_path, ddpm_astropt, device, args.ddim_steps, args.guidance_scale)
        torch.save(generated, args.output)
        plot_astropt_result(original, generated[0, 0].cpu().numpy(), metadata, args.save_png)
    
    print(f"Saved: {args.output}")