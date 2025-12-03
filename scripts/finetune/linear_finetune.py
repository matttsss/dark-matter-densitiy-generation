import torch, wandb, argparse

from tqdm.auto import tqdm
from dataclasses import asdict
from torch.utils.data import DataLoader
import torch.nn.functional as F

from scripts.embedings_utils import merge_datasets, compute_embeddings
from scripts.model_utils import LinearRegression, batch_to_device, load_model

argparser = argparse.ArgumentParser(description="Fine-tune AstroPT model on new tasks.")
argparser.add_argument("--pretrained_path", type=str, default="model/ckpt.pt",
                       help="Path to the pretrained model weights.")
argparser.add_argument("--output_path", type=str, default="model/finetuned_ckpt.pt",
                       help="Path to save the finetuned model weights.")
argparser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for training.")
argparser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for the optimizer.")
argparser.add_argument("--use_wandb", action="store_true",
                       help="Whether to log training metrics to Weights & Biases.")
argparser.add_argument("--num_epochs", type=int, default=60,
                       help="Number of epochs to train.")
argparser.add_argument("--label_names", type=str, nargs='+', default=["mass", "label", "BCG_e1", "BCG_e2", "BCG_stellar_conc"],
                       help="List of label names for the task.")
args = argparser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
weight_decay = 1e-4
lora_r = 0  # LoRA rank
betas = (0.9, 0.999)
num_epochs = args.num_epochs
label_names = args.label_names

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

if args.use_wandb:
    wandb_run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="matttsss-epfl",
        # Set the wandb project where this run will be logged.
        project="astropt_finetune",
        name="Linear Probe Finetune 5 labels",
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

# Load datasets
dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl",
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl",
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
        .select_columns(["images", "images_positions", *label_names]) \
        .shuffle(seed=42) \
        .map(lambda row:  {
            "images": row["images"],
            "images_positions": row["images_positions"],
            "labels": torch.stack(
                [(torch.log(row[label]) if label == "lensing_norm" else row[label]) 
                  for label in label_names], dim=0)
        })

dataset = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

train_dl = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    shuffle=True
)

val_dl = DataLoader(
    val_dataset,
    batch_size=4*batch_size,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    shuffle=False
)

# Load pretrained model
model = load_model(args.pretrained_path, device, strict=True)

# Train all parameters
for param in model.parameters():
    param.requires_grad = True

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay=weight_decay, learning_rate=learning_rate, 
    betas=betas, device_type=device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
    T_max=num_epochs, eta_min=1e-7, last_epoch=-1)


epoch_embedings, epoch_labels = compute_embeddings(
    model, train_dl, device, label_names, disable_tqdm=True)
epoch_labels = torch.stack([epoch_labels[label] for label in label_names], dim=1)

lin_reg = LinearRegression(device=device)
_ = lin_reg.fit(epoch_embedings, epoch_labels)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):

    model.train()
    train_loss = 0
    for B in tqdm(train_dl, 
                    desc=f"Training epoch {epoch}",
                    disable=wandb_run is not None):
        B = batch_to_device(B, device)

        batch_embeddings = torch.mean(model.get_embeddings(B)["images"], dim=1)
        predictions = lin_reg.predict(batch_embeddings)

        loss = F.mse_loss(predictions, B["labels"])
        loss.backward()

        train_loss += loss.item() / len(train_dl)

    model.eval()
    val_loss = 0
    with torch.no_grad():

        for B in tqdm(val_dl, 
                        desc=f"Validating epoch {epoch}",
                        disable=wandb_run is not None):
            B = batch_to_device(B, device)

            batch_embeddings = torch.mean(model.get_embeddings(B)["images"], dim=1)
            predictions = lin_reg.predict(batch_embeddings)
            
            loss = F.mse_loss(predictions, B["labels"])
            val_loss += loss.item() / len(val_dl)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    with torch.no_grad():
        epoch_embedings, epoch_labels = compute_embeddings(
            model, train_dl, device, label_names, disable_tqdm=wandb_run is not None)
        epoch_labels = torch.stack([epoch_labels[label] for label in label_names], dim=1)
        _ = lin_reg.fit(epoch_embedings, epoch_labels)

    print(f"Epoch {epoch}: val_loss {val_loss:.4f} and train_loss {train_loss:.4f}\n\n")
    if wandb_run is not None:
        wandb_run.log({
            "val_loss": val_loss, 
            "train_loss": train_loss, 
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model': model.state_dict(),
            'model_args': asdict(model.config),
            'modality_registry': model.modality_registry,
            'val_loss': best_val_loss,
            'train_loss': train_loss,
            'target_labels': label_names
        }, args.output_path)

if wandb_run is not None:
    wandb_run.finish()