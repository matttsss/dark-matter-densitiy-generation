import torch
from dataclasses import asdict

from torch.utils.data import DataLoader
from astropt.model_utils import load_astropt

from model_utils import LinearRegression
from model_utils import batch_to_device
from embedings_utils import merge_datasets, compute_embeddings
from utils import tqdm

# Config
pretrained_path = ("model/finetuned_weights/", "best_model.pt")
out_dir = "model/finetuned_weights"
batch_size = 16
learning_rate = 1e-5
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

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

labels = ["BCG_e1", "BCG_e2", "BCG_stellar_conc", "mass", "projection", "lensing_norm", "label"] # "norms", "redshift",
problematic_labels = torch.tensor([i for i, label in enumerate(labels) if label in ["lensing_norm"]])

# Load pretrained model
model = load_astropt(
    repo_id=None,
    path=pretrained_path[0],
    weights_filename=pretrained_path[1],
).to(device)

# Optimizer
optimizer = model.configure_optimizers(weight_decay=1e-5, learning_rate=learning_rate, betas=(0.9, 0.999), device_type=device)
criterion = torch.nn.HuberLoss()

val_losses = [[]]
train_losses = [[]]

with torch.no_grad():
    train_embeddings, train_labels = compute_embeddings(model, train_dl, device, labels)
    train_labels = torch.stack([train_labels[label] for label in labels], dim=1)
    lin_reg = LinearRegression(device=device)
    lin_reg.fit(train_embeddings, train_labels)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Train
    model.train()

    print(f"Training epoch {epoch}...")
    epoch_train_loss = 0
    for B in tqdm(train_dl):

        B = batch_to_device(B, device)
        batch_labels = torch.stack([B[label] for label in labels], dim=1)
        batch_labels[:, problematic_labels] = torch.log(batch_labels[:, problematic_labels])
        batch_embeddings = model.generate_embeddings(B)["images"]

        preds = lin_reg.predict(batch_embeddings)
        preds[:, problematic_labels] = torch.log(preds[:, problematic_labels])
        loss = criterion(preds, batch_labels)

        loss.backward(retain_graph=False)

        epoch_train_loss += loss.item() / len(train_dl)

            
    optimizer.step()
    optimizer.zero_grad()

    print(f"Computing new embeddings for epoch {epoch}...")
    with torch.no_grad():
        train_embeddings, train_labels = compute_embeddings(model, train_dl, device, labels)
        train_labels = torch.stack([train_labels[label] for label in labels], dim=1)
        lin_reg.fit(train_embeddings, train_labels)

    print(f"Validating epoch {epoch}...")
    # Validate
    model.eval()
    with torch.no_grad():
        embeddings_val, labels_val = compute_embeddings(model, val_dl, device, labels)
        labels_val = torch.stack([labels_val[label] for label in labels], dim=1)
        labels_val[:, problematic_labels] = torch.log(labels_val[:, problematic_labels])

        preds = lin_reg.predict(embeddings_val)
        preds[:, problematic_labels] = torch.log(preds[:, problematic_labels])
        val_loss = criterion(preds, labels_val)
        
        val_losses.append(val_loss.item())


    print(f"Epoch {epoch}: val_loss {val_loss:.4f} and train_loss {epoch_train_loss:.4f}\n\n")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model': model.state_dict(),
            'model_args': asdict(model.config),
            'modality_registry': model.modality_registry,
            'val_loss': best_val_loss,
            'epoch_train_loss': epoch_train_loss, 
        }, f"{out_dir}/best_model.pt")