# scripts/diffusion_train.py

import copy
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb

from model_utils import load_astropt_model
from embedings_utils import merge_datasets, compute_embeddings
from generative_model.DDPM import DDPM

LABEL_NAMES = ["mass", "label"]


def training_script(output_dir: str, weights_path: str):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Training diffusion on device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    run = wandb.init(
        entity="matttsss-epfl",
        project="astropt_diffusion",
        name="Diffusion v2 - patch4 cosine cfg",
    )

    # -------------------------------------------------
    # 1) Load finetuned AstroPT encoder
    # -------------------------------------------------
    print("Loading finetuned AstroPT checkpoint...")
    model = load_astropt_model(checkpoint_path=weights_path, device=device)
    model.eval()

    # -------------------------------------------------
    # 2) Load datasets and compute embeddings
    # -------------------------------------------------
    print("Loading datasets...")
    
    dataset = (
        merge_datasets([
            "data/BAHAMAS/bahamas_0.1.pkl",
            "data/BAHAMAS/bahamas_0.3.pkl",
            "data/BAHAMAS/bahamas_1.pkl",
            "data/BAHAMAS/bahamas_cdm.pkl",
        ])
        .select_columns(["images", "images_positions", *LABEL_NAMES])
        .shuffle(seed=42)
    )

    dataset_images = (
        merge_datasets([
            "data/BAHAMAS/bahamas_0.1.pkl",
            "data/BAHAMAS/bahamas_0.3.pkl",
            "data/BAHAMAS/bahamas_1.pkl",
            "data/BAHAMAS/bahamas_cdm.pkl",
        ], image_only=True)
        .shuffle(seed=42)
    )

    print(f"Dataset size: {len(dataset)}")

    # Compute embeddings
    print("Computing embeddings...")
    dl = DataLoader(dataset, batch_size=32, num_workers=4, prefetch_factor=3)
    embeddings, lab_dict = compute_embeddings(model, dl, device, LABEL_NAMES)
    embeddings = embeddings.cpu()
    
    print("Pre-loading images...")
    all_images = []
    for i in tqdm(range(len(dataset_images)), desc="Loading images"):
        img = np.array(dataset_images[i]["image"])
        all_images.append(img)
    all_images = torch.from_numpy(np.stack(all_images)).float()

    print(f"Raw images shape: {all_images.shape}")

    # Handle format - images are (N, C, H, W) already if shape[1] == 1
    if all_images.dim() == 4 and all_images.shape[1] != 1:
        # (N, H, W, C) -> (N, C, H, W)
        all_images = all_images.permute(0, 3, 1, 2)
    elif all_images.dim() == 3:
        # (N, H, W) -> (N, 1, H, W)
        all_images = all_images.unsqueeze(1)

    print(f"Final images shape: {all_images.shape}")
    
    print(f"Embeddings shape: {embeddings.shape}")

    # -------------------------------------------------
    # 3) Train/val split with combined dataset
    # -------------------------------------------------
    train_size = int(0.8 * len(embeddings))
    val_size = len(embeddings) - train_size
    
    # Split indices
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(embeddings), generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create train/val datasets
    train_dataset = TensorDataset(
        all_images[train_indices],
        embeddings[train_indices],
    )
    val_dataset = TensorDataset(
        all_images[val_indices],
        embeddings[val_indices],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Free memory
    del model, dataset, dataset_images, all_images, embeddings
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -------------------------------------------------
    # 4) Create DDPM
    # -------------------------------------------------
    print("Creating DDPM model...")
    
    diffusion_model = DDPM(
        patch_size=4,
        schedule="cosine",
    ).to(device)
    
    ema_model = copy.deepcopy(diffusion_model)
    ema_decay = 0.9999

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=3e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    epochs = 100
    best_val_loss = float("inf")
    cfg_dropout = 0.1

    # -------------------------------------------------
    # 5) Training loop
    # -------------------------------------------------
    print("Starting training...")
    
    for epoch in range(epochs):
        diffusion_model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, conditions in pbar:
            images = images.to(device)
            conditions = conditions.to(device)
            
            # CFG dropout
            if torch.rand(1).item() < cfg_dropout:
                conditions = torch.zeros_like(conditions)

            B = images.size(0)
            t = torch.randint(0, diffusion_model.timesteps, (B,), dtype=torch.long, device=device)

            noise = torch.randn_like(images)
            x_t = diffusion_model.q_sample(images, t, noise)
            noise_pred = diffusion_model.eps_model(x_t, t, conditions)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # EMA update
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), diffusion_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        lr_scheduler.step()

        # Validation
        ema_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, conditions in val_loader:
                images = images.to(device)
                conditions = conditions.to(device)

                B = images.size(0)
                t = torch.randint(0, ema_model.timesteps, (B,), dtype=torch.long, device=device)

                noise = torch.randn_like(images)
                x_t = ema_model.q_sample(images, t, noise)
                noise_pred = ema_model.eps_model(x_t, t, conditions)

                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        run.log({
            "epoch": epoch,
            "loss_train": train_loss,
            "loss_val": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model (val_loss={val_loss:.4f}), saving...")
            torch.save(
                ema_model.state_dict(),
                os.path.join(output_dir, "best_diffusion_model.pt"),
            )

    # Save final model
    torch.save(
        ema_model.state_dict(),
        os.path.join(output_dir, "final_diffusion_model_1000.pt"),
    )
    
    run.finish()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="model_weights/")
    parser.add_argument("--weights_path", type=str, default="model/finetuned_contrastive_ckpt.pt")
    args = parser.parse_args()

    training_script(args.output_dir, args.weights_path)