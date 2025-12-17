import torch, wandb, argparse
import numpy as np
from tqdm.auto import tqdm
from dataclasses import asdict
from torch.utils.data import DataLoader
import torch.nn.functional as F

from scripts.embedings_utils import merge_datasets, normalize_features
from scripts.model_utils import LinearRegression, batch_to_device, compute_embeddings, load_astropt_model


# ==============================
# Contrastive loss utilities
# ==============================

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    Supervised contrastive loss (Khosla et al. 2020 style) on a single label dimension.

    embeddings: [B, D]
    labels:     [B] (categorical or continuous; equality defines positives)
    """
    device = embeddings.device
    z = F.normalize(embeddings, dim=1)
    batch_size = z.size(0)

    if batch_size <= 1:
        return embeddings.new_tensor(0.0)

    labels = labels.contiguous().view(-1, 1)
    # mask[i, j] = 1 if same label, else 0
    mask = torch.eq(labels, labels.T).float().to(device)

    # similarity matrix
    logits = torch.matmul(z, z.T) / temperature

    # for numerical stability
    logits = logits - logits.max(dim=1, keepdim=True)[0]

    # mask out self-comparisons
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = mask.sum(dim=1)
    # avoid division by zero
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + 1e-12)

    # only anchors that have at least one positive
    valid = positive_count > 0
    if valid.sum() == 0:
        return embeddings.new_tensor(0.0)

    loss = -mean_log_prob_pos[valid].mean()
    return loss


def get_contrastive_weight(epoch, max_weight, warmup_epochs):
    """
    Linearly ramp contrastive weight from 0 to max_weight over warmup_epochs.
    After warmup, stays at max_weight.
    """
    if warmup_epochs <= 0 or max_weight <= 0:
        return 0.0
    if epoch >= warmup_epochs:
        return max_weight
    return max_weight * float(epoch + 1) / float(warmup_epochs)


# ==============================
# Main script
# ==============================

argparser = argparse.ArgumentParser(description="Fine-tune AstroPT model on new tasks (linear probe style).")
argparser.add_argument("--pretrained_path", type=str, default="model/ckpt.pt",
                       help="Path to the pretrained model weights.")
argparser.add_argument("--output_path", type=str, default="model/finetuned_contrastive_ckpt.pt",
                       help="Path to save the finetuned model weights.")
argparser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training.")
argparser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for the optimizer.")
argparser.add_argument("--use_wandb", action="store_true",
                       help="Whether to log training metrics to Weights & Biases.")
argparser.add_argument("--num_epochs", type=int, default=60,
                       help="Number of epochs to train.")
argparser.add_argument("--label_names", type=str, nargs='+',
                       default=["mass", "label", "BCG_e1", "BCG_e2", "BCG_stellar_conc"],
                       help="List of label names for the task.")
argparser.add_argument('--normalize_features', action='store_true',
                       help="Whether to normalize features.")

# contrastive-related
argparser.add_argument("--contrastive_weight", type=float, default=0.1,
                       help="Max weight for supervised contrastive loss on the 'label' dimension.")
argparser.add_argument("--contrastive_warmup_epochs", type=int, default=5,
                       help="Number of epochs to warm up contrastive weight from 0 to max.")
argparser.add_argument("--contrastive_temperature", type=float, default=0.1,
                       help="Temperature for supervised contrastive loss.")
args = argparser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
weight_decay = 1e-4
lora_r = 0  # LoRA rank (still unused here)
betas = (0.9, 0.999)
num_epochs = args.num_epochs
label_names = args.label_names

contrastive_max_w = args.contrastive_weight
contrastive_warmup_epochs = args.contrastive_warmup_epochs
contrastive_temperature = args.contrastive_temperature

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

if args.use_wandb:
    wandb_run = wandb.init(
        entity="matttsss-epfl",
        project="astropt_finetune",
        name="Linear Probe Finetune + SupCon",
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
            "contrastive_weight_max": contrastive_max_w,
            "contrastive_warmup_epochs": contrastive_warmup_epochs,
            "contrastive_temperature": contrastive_temperature,
        },
    )
else:
    wandb_run = None

# ----------------------------------------------------------
# Load datasets
# ----------------------------------------------------------
dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl",
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl",
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"], 
        label_names, stack_features=True)

dataset = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

if args.normalize_features:
    train_dataset, val_dataset = normalize_features(train_dataset, ["labels"], val_dataset=val_dataset)

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
    batch_size=4 * batch_size,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True,
    shuffle=False
)

# ----------------------------------------------------------
# Load pretrained model
# ----------------------------------------------------------
model = load_astropt_model(args.pretrained_path, device, strict=True)

# Train all parameters
for param in model.parameters():
    param.requires_grad = True

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay=weight_decay, learning_rate=learning_rate,
    betas=betas, device_type=device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-7, last_epoch=-1
)

# ----------------------------------------------------------
# Initial linear probe fit (on frozen embeddings)
# ----------------------------------------------------------
epoch_embedings, epoch_labels = compute_embeddings(
    model, train_dl, device, label_names, disable_tqdm=True)
epoch_labels_stacked = torch.stack(
    [epoch_labels[label] for label in label_names], dim=1
)

lin_reg = LinearRegression(device=device)
_ = lin_reg.fit(epoch_embedings, epoch_labels_stacked)

# Index of "label" inside stacked labels (for supervised contrastive)
label_idx_for_contrastive = None
if "label" in label_names:
    label_idx_for_contrastive = label_names.index("label")
if "log_label" in label_names:
    if label_idx_for_contrastive is not None:
        raise ValueError("Cannot use both 'label' and 'log_label' for contrastive loss.")
    label_idx_for_contrastive = label_names.index("log_label")

# ----------------------------------------------------------
# Training loop
# ----------------------------------------------------------
best_val_loss = float('inf')
for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    train_contrastive_loss = 0.0
    contrastive_weight = get_contrastive_weight(
        epoch, contrastive_max_w, contrastive_warmup_epochs
    )

    print(f"\nEpoch {epoch} / {num_epochs - 1} - using contrastive_weight={contrastive_weight:.4f}")

    # --------- TRAIN ----------
    for B in tqdm(train_dl,
                  desc=f"Training epoch {epoch}",
                  disable=wandb_run is not None):
        B = batch_to_device(B, device)

        # [B, D] embeddings
        batch_embeddings = torch.mean(model.get_embeddings(B)["images"], dim=1)
        predictions = lin_reg.predict(batch_embeddings)

        # MSE loss over all labels
        mse_loss = F.mse_loss(predictions, B["labels"])

        # Optional supervised contrastive loss on "label"
        if contrastive_weight > 0.0 and label_idx_for_contrastive is not None:
            contrastive_labels = B["labels"][:, label_idx_for_contrastive].detach()
            supcon_loss = supervised_contrastive_loss(
                batch_embeddings, contrastive_labels, temperature=contrastive_temperature
            )
        else:
            supcon_loss = batch_embeddings.new_tensor(0.0)

        total_loss = mse_loss + contrastive_weight * supcon_loss
        total_loss.backward()

        train_loss += mse_loss.item() * B["labels"].size(0)
        train_contrastive_loss += supcon_loss.item() * B["labels"].size(0)

    train_loss /= len(train_dataset)
    train_contrastive_loss /= len(train_dataset)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    # --------- VALIDATION ----------
    model.eval()
    val_loss = 0.0
    val_contrastive_loss = 0.0

    with torch.no_grad():
        for B in tqdm(val_dl,
                      desc=f"Validating epoch {epoch}",
                      disable=wandb_run is not None):
            B = batch_to_device(B, device)

            batch_embeddings = torch.mean(model.get_embeddings(B)["images"], dim=1)
            predictions = lin_reg.predict(batch_embeddings)

            mse_loss = F.mse_loss(predictions, B["labels"])

            if contrastive_weight > 0.0 and label_idx_for_contrastive is not None:
                contrastive_labels = B["labels"][:, label_idx_for_contrastive]
                supcon_loss = supervised_contrastive_loss(
                    batch_embeddings, contrastive_labels, temperature=contrastive_temperature
                )
            else:
                supcon_loss = batch_embeddings.new_tensor(0.0)

            val_loss += mse_loss.item() * B["labels"].size(0)
            val_contrastive_loss += supcon_loss.item() * B["labels"].size(0)

    val_loss /= len(val_dataset)
    val_contrastive_loss /= len(val_dataset)

    # --------- Refit linear probe on updated embeddings ----------
    with torch.no_grad():
        epoch_embedings, epoch_labels = compute_embeddings(
            model, train_dl, device, label_names, disable_tqdm=wandb_run is not None
        )
        epoch_labels_stacked = torch.stack(
            [epoch_labels[label] for label in label_names], dim=1
        )
        _ = lin_reg.fit(epoch_embedings, epoch_labels_stacked)

    print(
        f"Epoch {epoch}: "
        f"val_loss {val_loss:.4f} (con={val_contrastive_loss:.4f}) | "
        f"train_loss {train_loss:.4f} (con={train_contrastive_loss:.4f})"
    )

    if wandb_run is not None:
        wandb_run.log({
            "val_loss": val_loss,
            "train_loss": train_loss,
            "val_contrastive_loss": val_contrastive_loss,
            "train_contrastive_loss": train_contrastive_loss,
            "contrastive_weight": contrastive_weight,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

    # Save best model (based on *MSE* val_loss, not including contrastive)
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
        print(f"  -> Saved new best model with val_loss={best_val_loss:.4f}")

if wandb_run is not None:
    wandb_run.finish()