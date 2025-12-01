import torch, wandb
from dataclasses import asdict

from torch.utils.data import DataLoader

from model_utils import load_model, batch_to_device
from embedings_utils import merge_datasets

# Config
pretrained_path = "model/ckpt.pt"
out_dir = "model/finetuned_weights"
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-3
lora_r = 8  # LoRA rank
betas = (0.9, 0.999)
num_epochs = 20
labels = ["BCG_e1", "BCG_e2", "BCG_stellar_conc", "mass", "label"]

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="matttsss-epfl",
    # Set the wandb project where this run will be logged.
    project="astropt_finetune",
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
        "labels": labels,
        "epochs": num_epochs,
    },
)

# Load datasets
dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl",
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl",
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"])

dataset = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

train_dl = DataLoader(
    train_dataset,
    batch_size=batch_size,
    pin_memory=True,
    shuffle=True
)

val_dl = DataLoader(
    val_dataset,
    batch_size=128,
    pin_memory=True,
    shuffle=False
)

# Load pretrained model
model = load_model(pretrained_path, device, lora_r=lora_r, output_dim=len(labels))

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay=weight_decay, learning_rate=learning_rate, 
    betas=betas, device_type=device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
    T_max=num_epochs, eta_min=0, last_epoch=-1)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):

    model.train()
    print(f"Training epoch {epoch}...")
    epoch_train_loss = 0
    for B in train_dl:

        B = batch_to_device(B, device)
        batch_labels = torch.stack([B[label] for label in labels], dim=1)
        res, loss = model.get_task_prediction(B, targets=batch_labels)

        loss.backward(retain_graph=False)

        epoch_train_loss += loss.item() / len(train_dl)

            
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    model.eval()
    print(f"Validating epoch {epoch}...")
    epoch_val_loss = 0
    with torch.no_grad():

        for B in val_dl:

            B = batch_to_device(B, device)
            batch_labels = torch.stack([B[label] for label in labels], dim=1)
            res, loss = model.get_task_prediction(B, targets=batch_labels)

            epoch_val_loss += loss.item() / len(val_dl)

    print(f"Epoch {epoch}: val_loss {epoch_val_loss:.4f} and train_loss {epoch_train_loss:.4f}\n\n")
    run.log({"val_loss": epoch_val_loss, "train_loss": epoch_train_loss, "learning_rate": optimizer.param_groups[0]['lr']})

    # Save best model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save({
            'model': model.state_dict(),
            'model_args': asdict(model.config),
            'modality_registry': model.modality_registry,
            'val_loss': best_val_loss,
            'epoch_train_loss': epoch_train_loss, 
        }, f"{out_dir}/best_model.pt")

wandb.finish()