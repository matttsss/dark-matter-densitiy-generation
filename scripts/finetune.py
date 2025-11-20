import torch
import loralib as lora

from astropt.model import GPT
from astropt.local_datasets import GalaxyImageDataset

from embedings_utils import get_embeddings

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Config
pretrained_path = "model/ckpt.pt"
out_dir = "model/finetuned_weights"
batch_size = 32
learning_rate = 1e-4
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load pretrained model
checkpoint = torch.load(pretrained_path)
config = checkpoint["config"]

# Add finetuning configs
config.lora_r = 8  # LoRA rank
config.output_dim = 1  # Your task dimension

# Initialize model

model = GPT(config).to(device)
model.load_state_dict(checkpoint["model"])

# Setup LoRA
lora.mark_only_lora_as_trainable(model)
for param in model.task_head.parameters():
    param.requires_grad = True

# Load datasets
train_dataset = GalaxyImageDataset("train_data.txt")
val_dataset = GalaxyImageDataset("val_data.txt")

# Optimizer
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Train
    model.train()
    optimizer.zero_grad()

    # Change target_list to list of available labels we can do the regression on. 

    embeddings_train, labels_train = get_embeddings(model, train_dataset, labels=target_list)

    val_losses = [[]]
    train_losses = [[]]


    labels = labels_train[target]


    lin_reg = LinearRegression()
    lin_reg.fit(embeddings_train, labels)
    preds = lin_reg.predict(embeddings_train)

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