import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from scripts.model_utils import load_model
from scripts.embedings_utils import merge_datasets, compute_embeddings

class VectorField(nn.Module):
    def __init__(self, dim, nb_conditions, hidden=256):
        super().__init__()

        self.dim = dim
        self.nb_conditions = nb_conditions
    
        self.net = nn.Sequential(
            nn.Linear(dim + nb_conditions + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, cond, t):
        """
        x: [B, D] state
        t: [B, 1] time in [0,1]
        """
        return self.net(torch.cat([x, cond, t], dim=-1))


def flow_matching_loss(v_theta: VectorField, x0, x1, cond):
    """
    x0: [B, D] samples from source distribution
    x1: [B, D] samples from target distribution (e.g. dataset)
    """
    B = x0.size(0)

    # sample time uniformly
    t = torch.rand(B, 1, device=x0.device)

    # interpolate between x0 and x1
    x_t = (1 - t) * x0 + t * x1

    # target velocity (CFM)
    v_target = x1 - x0

    # predict vector field
    v_pred = v_theta(x_t, cond, t)

    return F.mse_loss(v_pred, v_target)


# ----------------------------------------------------
# 3. ODE Sampling
# ----------------------------------------------------
@torch.no_grad()
def sample_flow(v_theta: VectorField, cond, steps=100):
    """
    Integrates dx/dt = v_theta(x, t) from t=0 to 1.
    Start from simple noise distribution.
    """
    x = torch.randn(cond.size(0), v_theta.dim).to(next(v_theta.parameters()).device)
    dt = 1.0 / steps

    t = torch.zeros(cond.size(0), 1).to(x.device)
    for _ in range(steps):
        v = v_theta(x, cond, t)
        x = x + v * dt
        t = t + dt

    return x


# ----------------------------------------------------
# 4. Training Loop Example
# ----------------------------------------------------
def train_flow_matching(
        train_embed_dl: DataLoader, val_embed_dl: DataLoader, 
        device, embed_dim, nb_cond, epochs=10, lr=1e-3):

    v_theta = VectorField(embed_dim, nb_cond).to(device)
    opt = torch.optim.AdamW(v_theta.parameters(), lr=lr)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = 0.0
        for x1, cond in train_embed_dl:
            x1 = x1.to(device).view(x1.size(0), -1)
            cond = cond.to(device).view(cond.size(0), -1)

            # source distribution = standard Gaussian
            x0 = torch.randn_like(x1)

            loss = flow_matching_loss(v_theta, x0, x1, cond)
            opt.zero_grad()
            loss.backward()
            opt.step()
                
            train_loss += loss.item()

        train_loss /= len(train_embed_dl)

        val_loss = 0.0
        for x1, cond in val_embed_dl:
            x1 = x1.to(device).view(x1.size(0), -1)
            cond = cond.to(device).view(cond.size(0), -1)

            sampled_val_embed = sample_flow(v_theta, cond)
            val_loss += F.mse_loss(sampled_val_embed, x1).item()

        val_loss /= len(val_embed_dl)

        print(f"Epoch {epoch+1}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(v_theta.state_dict(), "model/flow_matching/best_fm_model.pt")

    return v_theta


# ----------------------------------------------------
# Example Usage (pseudo-code)
# ----------------------------------------------------
if __name__ == "__main__":
    device = ("mps" if torch.backends.mps.is_available() else 
              "cuda" if torch.cuda.is_available() else 
              "cpu")
    parser = argparse.ArgumentParser(
                    prog='GenAstroPT',
                    description='Trains a flow matching model on astroPT embeddings')
    parser.add_argument('--nb_points', type=int, default=1000, help='Number of points to use for embeddings')
    parser.add_argument('--labels', nargs='+', default=["mass", "label"], help='Labels to use for the conditions of the flow matching model')
    parser.add_argument('--model_path', type=str, default="model/ckpt.pt", help='Path to the astropt checkpoint')
    args = parser.parse_args()
    
    labels_name = args.labels  

    has_metals = torch.backends.mps.is_available()  
    device = torch.device('mps' if has_metals else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    
    print(f"Generating embeddings on device: {device}")
    model = load_model(args.model_path, device=device, strict=True)
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
            .select_columns(["images", "images_positions", *labels_name]) \
            .shuffle(seed=42) \
            .take(args.nb_points)    

    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 128,
        num_workers = 0 if has_metals else 4,
        prefetch_factor = None if has_metals else 3
    )

    embeddings, cond = compute_embeddings(model, dl, device, labels_name)

    embeddings = embeddings.cpu()
    cond = {k: v.cpu() for k, v in cond.items()}
    cond = torch.stack([cond[k] for k in labels_name], dim=-1)

    # Split into train and val
    nb_train = int(0.8 * embeddings.size(0))
    train_embed_dl = DataLoader(TensorDataset(embeddings[:nb_train], cond[:nb_train]), 
                                batch_size=512, shuffle=True)
    val_embed_dl = DataLoader(TensorDataset(embeddings[nb_train:], cond[nb_train:]), 
                              batch_size=512, shuffle=False)
    
    v_theta = train_flow_matching(train_embed_dl, val_embed_dl, device,
                                  embed_dim=embeddings.size(-1), 
                                  nb_cond=cond.size(-1), epochs=50, lr=1e-3)
