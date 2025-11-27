import torch, wandb
from dataclasses import asdict

import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_utils import load_model, batch_to_device
from embedings_utils import merge_datasets

# Config
pretrained_path = "model/ckpt.pt"
batch_size = 150
learning_rate = 1e-3
weight_decay = 1e-3
lora_r = 0  # LoRA rank
betas = (0.9, 0.999)
num_epochs = 50
label_names = ["BCG_e1", "BCG_e2", "BCG_stellar_conc", "mass", "label"]

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

if True:
    wandb_run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="matttsss-epfl",
        # Set the wandb project where this run will be logged.
        project="astropt_finetune_head",
        name="Hubber Loss Cosine Annealing LR",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "architecture": "AstroPT",
            "dataset": "BAHAMAS 4/6",
            "batch_size": batch_size,
            "betas": betas,
            "lora_r": lora_r,
            "labels": label_names,
            "epochs": num_epochs,
        },
    )
else:
    wandb_run = None

print(f"Using device: {device}")

def merge_columns(row):
    merged_labels = torch.stack([row[label] for label in label_names], dim=0)
    return {
        "images": row["images"],
        "images_positions": row["images_positions"],
        "labels": merged_labels
    }

# Load datasets
dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl",
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl",
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
        .select_columns(["images", "images_positions", *label_names]) \
        .map(merge_columns)

dataset = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

train_dl = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=2,
    prefetch_factor=2,
    pin_memory=True,
    shuffle=True
)

val_dl = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=2,
    prefetch_factor=2,
    pin_memory=True,
    shuffle=False
)

# Load pretrained model
model = load_model(pretrained_path, device, lora_r, output_dim=len(label_names))
print(f"Loaded model from {pretrained_path}")

# Freeze all model parameters except LoRA and task head
for p in model.parameters(): p.requires_grad = False
for p in model.task_head.parameters(): p.requires_grad = True
# lora.mark_only_lora_as_trainable(model)

optimizer = model.configure_optimizers(
    weight_decay=weight_decay, learning_rate=learning_rate, 
    betas=betas, device_type=device
)

# ===============================================================
# =================== Run head optimization =====================
# ===============================================================
best_val_loss = float('inf')
for epoch in range(num_epochs):
    
    model.train()
    print(f"Training epoch {epoch}...")
    
    train_loss = 0
    for B in train_dl:
        B = batch_to_device(B, device)
        with torch.no_grad():
            embeddings = model.get_embeddings(B)["images"]
        
        res = model.task_head(embeddings)
        loss = F.huber_loss(res, B["labels"])

        loss.backward()

        train_loss += loss.item() / len(train_dl)

    optimizer.step()
    optimizer.zero_grad()

    model.eval()
    print(f"Validating epoch {epoch}...")
    
    val_loss = 0
    with torch.no_grad():

        for B in val_dl:
            B = batch_to_device(B, device)
            with torch.no_grad():
                embeddings = model.get_embeddings(B)["images"]

            res = model.task_head(embeddings)
            loss = F.huber_loss(res, B["labels"])

            val_loss += loss.item() / len(val_dl)


    print(f"Epoch {epoch}: val_loss {val_loss:.4f} and train_loss {train_loss:.4f}\n\n")
    if wandb_run is not None:
        wandb_run.log({
            "head_train_loss": train_loss,
            "head_val_loss": val_loss,
        })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model': model.state_dict(),
            'model_args': asdict(model.config),
            'modality_registry': model.modality_registry,
            'val_loss': best_val_loss,
        }, f"model/finetuned_head.pt")

wandb_run.finish()
