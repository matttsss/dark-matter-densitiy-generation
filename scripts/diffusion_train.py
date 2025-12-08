# scripts/diffusion_train.py

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn.functional as F
import numpy as np  
import wandb

from model_utils import load_astropt_model
from embedings_utils import merge_datasets, compute_embeddings
from generative_model.DDPM import DDPM

# Labels used as conditions
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
        name="Diffusion FIRST TRUE RUN",
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

    model = load_astropt_model(
        checkpoint_path=weights_path,
        device=device
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

    
    max_points = len(dataset)
    nb_points_effective = max_points
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

    # Conditions = [mass, label]
    dataset_embeddings = TensorDataset(
        embeddings, lab_dict["mass"], lab_dict["label"]
    )

    generator = torch.Generator().manual_seed(42)

    train_size = int(0.8 * len(dataset_embeddings))
    val_size = len(dataset_embeddings) - train_size
    train_dataset, val_dataset = random_split(
        dataset_embeddings, [train_size, val_size],generator=generator
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
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    dataset_images_only = (
        merge_datasets([
            "data/BAHAMAS/bahamas_0.1.pkl",
            "data/BAHAMAS/bahamas_0.3.pkl",
            "data/BAHAMAS/bahamas_1.pkl",
            "data/BAHAMAS/bahamas_cdm.pkl",
        ], image_only=True)
        .shuffle(seed=42)
    )
    

    dataset_images_only_train = dataset_images_only.select(range(train_size))
    dataset_images_only_val = dataset_images_only.select(range(train_size, len(dataset_images_only)))


    dataloader_images_only_train = DataLoader(
        dataset_images_only_train,
        batch_size=128,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    dataloader_images_only_val = DataLoader(
        dataset_images_only_val,
        batch_size=128,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------
    # 3) Create DDPM over embedding space
    # -------------------------------------------------
    print(f"Training DDPM")

    diffusion_model = DDPM().to(device)

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

        for batch, images_dict in zip(train_loader,dataloader_images_only_train):
            
            images_array = np.array(images_dict["image"])
            x_0 = torch.from_numpy(images_array).float().to(device)
            x_0 = x_0.permute(3, 0, 1, 2)

            conditions = batch[0].to(device)

            B = x_0.size(0)
            t = torch.randint(
                0, diffusion_model.timesteps,
                (B,), dtype=torch.long, device=device
            )

            noise = torch.randn_like(x_0)
            x_t = diffusion_model.q_sample(x_0, t, noise)
            noise_pred = diffusion_model.eps_model(x_t, t, conditions)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

        train_loss /= max(1, num_train_batches)
        lr_scheduler.step()

        # -----------------------------
        # Validation
        # -----------------------------
        diffusion_model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch, images_dict in zip(val_loader,dataloader_images_only_val):

                images_array = np.array(images_dict["image"])
                x_0 = torch.from_numpy(images_array).float().to(device)
                x_0 = x_0.permute(3, 0, 1, 2)
      
                conditions = batch[0].to(device)

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
                wandb.log({
                "batch_loss": loss.item(),
            })

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
            print("saved best diffusion model")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,default="model_weights/")
    parser.add_argument(
        "--weights_path",
        type=str,
        default="model/finetuned_contrastive_ckpt.pt",
    )

    args = parser.parse_args()

    training_script(args.output_dir, args.weights_path)