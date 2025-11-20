from tqdm.auto import tqdm
import torch

from torch.utils.data import DataLoader
from astropt.model_utils import load_astropt

from model_utils import LinearRegression
from model_utils import batch_to_device
from embedings_utils import merge_datasets, get_embeddings

# Config
pretrained_path = "model/ckpt.pt"
out_dir = "model/finetuned_weights"
batch_size = 128
learning_rate = 1e-4
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load pretrained model
model = load_astropt(
    repo_id=None,
    path="model/",
    weights_filename="ckpt.pt",
).to(device)

for param in model.parameters():
    param.requires_grad = True

# Load datasets
dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl",
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl",
        "data/DarkData/BAHAMAS/bahamas_1.pkl",])

dataset = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = dataset['train']
val_test_dataset = dataset['test'].train_test_split(test_size=0.5, seed=42)
val_dataset = val_test_dataset['train']
test_dataset = val_test_dataset['test']

train_dl = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    prefetch_factor=3
)

val_dl = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)

labels = ["BCG_e1", "BCG_e2", "BCG_stellar_conc", "mass", "lensing_norm", "label", "norms"]

# Optimizer
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

val_losses = [[]]
train_losses = [[]]

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Train
    model.train()
    optimizer.zero_grad()

    train_embeddings = []
    train_labels = {label: [] for label in labels}
    for B in tqdm(train_dl):

        B = batch_to_device(B, device)
        batch_embeddings = model.get_embeddings(B)["images"]

        train_embeddings.append(batch_embeddings)

        for label in labels:
            train_labels[label].append(B[label])

    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = [torch.cat(train_labels[label], dim=0) for label in labels]
    train_labels = torch.stack(train_labels, dim=1)


    lin_reg = LinearRegression()
    lin_reg.fit(train_embeddings, train_labels)

    train_loss = mean_squared_error(labels, preds)
    epoch_train_loss = mean_squared_error(labels, preds,multioutput='raw_values')
    loss_tensor = torch.tensor(train_loss, requires_grad=True).to(device)
    train_losses.append(epoch_train_loss)

    
    loss_tensor.backward()
    optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        embeddings_val, labels_val = get_embeddings(model, val_dataset, labels=["target"])
        
        labels = labels_val[target]


        preds = lin_reg.predict(embeddings_val)


        val_loss = mean_squared_error(labels, preds)
        epoch_val_loss = mean_squared_error(labels, preds,multioutput='raw_values')

        
        val_losses.append(epoch_val_loss)


    print(f"Epoch {epoch}: val_loss {val_loss:.4f} and train_loss {train_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'model': model.state_dict(),
            'config': config,
            'val_loss': best_val_loss,
            'detailed_loss': epoch_val_loss, 
        }, f"{out_dir}/best_model.pt")