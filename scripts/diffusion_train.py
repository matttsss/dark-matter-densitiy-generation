# scripts/diffusion_train.py

import argparse
<<<<<<< HEAD

from generative_model.DDPM import DDPM
from model_utils import load_model
from embedings_utils import merge_datasets, compute_embeddings, labels_name
=======
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

>>>>>>> 0055c06 (Add code for diffusion model)
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn.functional as F
import wandb

<<<<<<< HEAD
def training_script(output_dir = "../results_diffusion",weights_path = "../model/finetuned_contrastive_ckpt.pt"):
=======
from model_utils import load_model
from embedings_utils import merge_datasets, compute_embeddings
from generative_model.DDPM import DDPM

# Labels used as conditions
LABEL_NAMES = ["mass", "label"]


def training_script(output_dir: str, weights_path: str, nb_points: int = 10000):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Training diffusion on device: {device}")
>>>>>>> 0055c06 (Add code for diffusion model)

    os.makedirs(output_dir, exist_ok=True)

    run = wandb.init(
        entity="matttsss-epfl",
        project="astropt_diffusion",
        name="Diffusion first try",
    )

    # -------------------------------------------------
    # 1) Load finetuned AstroPT (contrastive) encoder
    # -------------------------------------------------
    print("Loading finetuned AstroPT checkpoint metadata...")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model_args = ckpt.get("model_args", {})

    embed_dim_meta = model_args.get("embed_dim", 384)   # may or may not be correct
    lora_rank = model_args.get("lora_r", 0)
    output_dim = model_args.get("output_dim", 0)

    print(f" -> (meta) embed_dim={embed_dim_meta}, lora_rank={lora_rank}, output_dim={output_dim}")

    model = load_model(
        checkpoint_path=weights_path,
        device=device,
        lora_rank=lora_rank,
        output_dim=output_dim,
    )
    model.eval()

    # -------------------------------------------------
    # 2) Build BAHAMAS dataset & compute embeddings
    # -------------------------------------------------
    labels_name = LABEL_NAMES

    dataset = (
        merge_datasets([
            "data/BAHAMAS/bahamas_0.1.pkl",
            "data/BAHAMAS/bahamas_0.3.pkl",
            "data/BAHAMAS/bahamas_1.pkl",
            "data/BAHAMAS/bahamas_cdm.pkl",
        ])
        .select_columns(["images", "images_positions", *labels_name])
        .shuffle(seed=42)
    )

    if nb_points is not None:
        max_points = len(dataset)
        nb_points_effective = min(nb_points, max_points)
        print(f"Using {nb_points_effective} samples out of {max_points} available")
        dataset = dataset.select(range(nb_points_effective))

    dl = DataLoader(
        dataset,
        batch_size=128,
        num_workers=4,
        prefetch_factor=3,
        pin_memory=(device.type == "cuda"),
    )

    # Compute AstroPT embeddings & labels
    embeddings, lab_dict = compute_embeddings(
        model, dl, device, LABEL_NAMES
    )
    embeddings = embeddings.cpu()
    lab_dict = {k: v.cpu() for k, v in lab_dict.items()}

    print(f"Embeddings shape: {embeddings.shape}")

    # Conditions = [mass, label]
    dataset_embeddings = TensorDataset(
        embeddings, lab_dict["mass"], lab_dict["label"]
    )

    train_size = int(0.8 * len(dataset_embeddings))
    val_size = len(dataset_embeddings) - train_size
    train_dataset, val_dataset = random_split(
        dataset_embeddings, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
<<<<<<< HEAD
        num_workers=4)

    
    diffusion_model = DDPM()
    diffusion_model.to(device)
    
    epochs = 50
=======
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------
    # 3) Create DDPM over embedding space
    # -------------------------------------------------
    embed_dim = embeddings.shape[1]
    print(f"Training DDPM in embedding space of dimension D={embed_dim}")

    diffusion_model = DDPM(
        data_dim=(embed_dim,),
        timesteps=200,
        cond_dim=2,        # [mass, label]
        dit_hidden_size=128,
        dit_depth=3,
        dit_heads=4,
    ).to(device)
>>>>>>> 0055c06 (Add code for diffusion model)

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=3e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50
    )

    epochs = 50
    best_val_loss = float("inf")

    for epoch in range(epochs):
        diffusion_model.train()
        train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            x_0 = batch[0].to(device)        # (B, D)
            mass = batch[1].to(device)       # (B,)
            label = batch[2].to(device)      # (B,)

<<<<<<< HEAD
        for batch in dataloader_train:
            x_0 = batch[0].to(device)
            conditions = batch[1:].to(device)
=======
            # Conditions: [mass, label]
            conditions = torch.stack([mass, label], dim=1).float()  # (B, 2)
>>>>>>> 0055c06 (Add code for diffusion model)

            B = x_0.size(0)
            t = torch.randint(
                0, diffusion_model.timesteps,
                (B,), dtype=torch.long, device=device
            )

            noise = torch.randn_like(x_0)
            x_t = diffusion_model.q_sample(x_0, t, noise)
            noise_pred = diffusion_model.eps_model(x_t, t, conditions)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

<<<<<<< HEAD
            loss_val = 0
            diffusion_model.eval()

            for batch in dataloader_val:
                x_0 = batch[0].to(device)
                conditions = batch[1:].to(device)
=======
        train_loss /= max(1, num_train_batches)
        lr_scheduler.step()
>>>>>>> 0055c06 (Add code for diffusion model)

        # -----------------------------
        # Validation
        # -----------------------------
        diffusion_model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x_0 = batch[0].to(device)
                mass = batch[1].to(device)
                label = batch[2].to(device)
                conditions = torch.stack([mass, label], dim=1).float()

                B = x_0.size(0)
                t = torch.randint(
                    0, diffusion_model.timesteps,
                    (B,), dtype=torch.long, device=device
                )

                noise = torch.randn_like(x_0)
                x_t = diffusion_model.q_sample(x_0, t, noise)
                noise_pred = diffusion_model.eps_model(x_t, t, conditions)

                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= max(1, num_val_batches)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )
        run.log({
            "epoch": epoch,
            "loss_train": train_loss,
            "loss_val": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best diffusion model...")
            torch.save(
                diffusion_model.state_dict(),
                os.path.join(output_dir, "best_diffusion_model.pt"),
            )

    run.finish()

<<<<<<< HEAD
    args = argparse.ArgumentParser()
    args.add_argument("--output_dir", type=str, help="Output directory to save the trained model")
    args.add_argument("--weights_path", type=str, default="", help="Path to pretrained weights")
    args = args.parse_args()
    
    training_script(args.output_dir, args.weights_path)
=======
>>>>>>> 0055c06 (Add code for diffusion model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--weights_path",
        type=str,
        default="model/finetuned_contrastive_ckpt.pt",
    )
    parser.add_argument("--nb_points", type=int, default=20000)
    args = parser.parse_args()

    training_script(args.output_dir, args.weights_path, nb_points=args.nb_points)