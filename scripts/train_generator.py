"""
Trains a flow matching model on astroPT embeddings.

python3 -m scripts.train_generator \
    --model_path <astropt_model_path> \
    --nb_points 14000 --epochs 5000 --sigma 1.0
"""

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, asdict, field
from torchcfm.conditional_flow_matching import *

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from scripts.model_utils import load_model
from scripts.plot_utils import plot_cross_section_histogram
from scripts.embedings_utils import merge_datasets, compute_embeddings

@dataclass
class VectorFieldConfig:
    sigma: float = 1.0
    dim: int = 128
    encoding_size: int = 64
    hidden: int = 512
    conditions: list[str] = field(default_factory=list)

class VectorField(nn.Module, TargetConditionalFlowMatcher):
    
    def __init__(self, config: VectorFieldConfig):
        nn.Module.__init__(self)
        TargetConditionalFlowMatcher.__init__(self, config.sigma)
    
        self.config = config
        self.ot_sampler = OTPlanSampler("exact", num_threads="max")

        self.encoding_net = nn.Sequential(
            nn.Linear(len(config.conditions), config.encoding_size//2),
            nn.SiLU(),
            nn.Linear(config.encoding_size//2, config.encoding_size),
            nn.SiLU()
        )
    
        self.net = nn.Sequential(
            nn.Linear(config.dim + config.encoding_size + 1, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, cond, t):
        encoding = self.encoding_net(cond)
        return self.net(torch.cat([x, encoding, t], dim=-1))


    def compute_loss(self, x0, x1, cond):
        t, xt, ut = self.sample_location_and_conditional_flow(x0, x1)
        return F.mse_loss(self(xt, cond, t.view(-1, 1)), ut)


@torch.no_grad()
def sample_flow(v_theta: VectorField, cond, steps=100):
    """
    Integrates dx/dt = v_theta(x, t) from t=0 to 1.
    Start from simple noise distribution.
    """
    dt = 1.0 / steps
    batch_size = cond.size(0)

    t = torch.zeros(batch_size, 1, device=cond.device)
    x = torch.randn(batch_size, v_theta.config.dim, device=cond.device)

    for _ in range(steps):
        x += dt * v_theta(x, cond, t)
        t += dt

    return x



def train_flow_matching(
        train_embed_dl: DataLoader, val_embed_dl: DataLoader, 
        device, vf_config: VectorFieldConfig, 
        model_name,
        epochs=10, lr=1e-3):
    
    v_theta = VectorField(vf_config).to(device)
    opt = torch.optim.AdamW(v_theta.parameters(), lr=lr)

    best_val_loss = float('inf')
    epoch_val_losses = []
    epoch_train_losses = []
    for epoch in range(epochs):
        train_loss = 0.0
        for x1, cond in train_embed_dl:
            x1 = x1.to(device).view(x1.size(0), -1)
            cond = cond.to(device).view(cond.size(0), -1)

            x0 = torch.randn_like(x1)
            loss = v_theta.compute_loss(x0, x1, cond)

            opt.zero_grad()
            loss.backward()
            opt.step()
                
            train_loss += loss.item()

        train_loss /= len(train_embed_dl)

        val_loss = 0.0
        for x1, cond in val_embed_dl:
            x1 = x1.to(device).view(x1.size(0), -1)
            cond = cond.to(device).view(cond.size(0), -1)

            x0 = torch.randn_like(x1)
            loss = v_theta.compute_loss(x0, x1, cond)

            val_loss += loss.item()

        val_loss /= len(val_embed_dl)
        
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                        "state_dict": v_theta.state_dict(),
                        "config": asdict(vf_config),
                        "conditions": vf_config.conditions
                    }, 
                    f"model/flow_matching/{model_name}.pt")

    return v_theta, epoch_train_losses, epoch_val_losses


# ----------------------------------------------------
# Example Usage (pseudo-code)
# ----------------------------------------------------
if __name__ == "__main__":
    from .probes.validate_fm_model import predict_fm_model, umap_compare

    device = ("mps" if torch.backends.mps.is_available() else 
              "cuda" if torch.cuda.is_available() else 
              "cpu")
    parser = argparse.ArgumentParser(
                    prog='GenAstroPT',
                    description='Trains a flow matching model on astroPT embeddings')
    parser.add_argument('--nb_points', type=int, default=1000, help='Number of points to use for embeddings')
    parser.add_argument('--labels', nargs='+', default=["mass", "label"], help='Labels to use for the conditions of the flow matching model')
    parser.add_argument('--model_path', type=str, default="model/ckpt.pt", help='Path to the astropt checkpoint')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--sigma', type=float, default=0.01, help='Sigma value for the flow matching model')
    args = parser.parse_args()
    
    has_metals = torch.backends.mps.is_available()  
    device = torch.device('mps' if has_metals else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    
    print(f"Generating embeddings on device: {device}")

    model_name = args.model_path.split('/')[-1].replace('.pt','')
    sigma_name = f"sigma_{args.sigma:.3f}".replace('.','_')

    model = load_model(args.model_path, device=device, strict=True)
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
            .select_columns(["images", "images_positions", *args.labels]) \
            .shuffle(seed=42) \
            .take(args.nb_points)    

    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 256,
        num_workers = 0 if has_metals else 4,
        prefetch_factor = None if has_metals else 3
    )

    embeddings, cond_dict = compute_embeddings(model, dl, device, args.labels)
    cond = torch.stack([cond_dict[k] for k in args.labels], dim=-1)

    # Split into train and val
    nb_train = int(0.8 * embeddings.size(0))
    train_embed_dl = DataLoader(TensorDataset(embeddings[:nb_train], cond[:nb_train]), 
                                batch_size=nb_train)
    val_embed_dl = DataLoader(TensorDataset(embeddings[nb_train:], cond[nb_train:]), 
                              batch_size=embeddings.size(0) - nb_train)

    vf_config = VectorFieldConfig(
        sigma=args.sigma,
        dim=embeddings.size(-1),
        hidden=512,
        conditions=args.labels
    )

    v_theta, epoch_train_losses, epoch_val_losses = \
        train_flow_matching(train_embed_dl, val_embed_dl, device, vf_config,
                            model_name=f"fm_{model_name}_{sigma_name}",
                            epochs=args.epochs, lr=1e-4)
    
    # ==============================================
    # Plot training curves
    # ==============================================

    cutoff = args.epochs//10
    x = range(cutoff, args.epochs)
    epoch_train_losses = epoch_train_losses[cutoff:]
    epoch_val_losses = epoch_val_losses[cutoff:]

    plt.plot(x, epoch_train_losses, label='Train Loss')
    plt.plot(x, epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Flow Matching Training and Validation Loss')
    plt.legend()
    plt.savefig('figures/flow_matching/loss_curve.png')
    plt.show()
    plt.close()

    train_ratio = 0.8
    train_nb = int(train_ratio * embeddings.shape[0])
    vf_preds_embeddings, vf_preds, lin_preds = \
        predict_fm_model(v_theta, embeddings, cond_dict, args.labels, args.labels, train_ratio=train_ratio)
    
    embeddings = embeddings.cpu().numpy()
    vf_preds_embeddings = vf_preds_embeddings.cpu().numpy()
    lin_preds = {k: v.cpu().numpy() for k, v in lin_preds.items()}
    vf_preds = {k: v.cpu().numpy() for k, v in vf_preds.items()}
    ground_truth = {k: v.cpu().numpy() for k, v in cond_dict.items()}

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    rel_diff = np.abs(lin_preds["mass"] - ground_truth["mass"]) / np.abs(ground_truth["mass"])
    axs[0,0].scatter(ground_truth["mass"], lin_preds["mass"], alpha=0.1, label='Linear Regression Predictions')
    res = axs[0,0].scatter(ground_truth["mass"], vf_preds["mass"], alpha=0.1, 
                     c=rel_diff, label='Flow Matching Predictions')
    fig.colorbar(res, ax=axs[0,0], label='Relative Difference')

    axs[0,0].set_title(f"Predictions for mass")
    axs[0,0].set_xlabel(f"Ground truth mass")
    axs[0,0].set_ylabel(f"Predicted mass")
    axs[0,0].legend()

    fig.delaxes(axs[0,1])

    plot_cross_section_histogram(axs[1, 0], ground_truth["label"], lin_preds["label"], "reference embeddings")
    plot_cross_section_histogram(axs[1, 1], ground_truth["label"], vf_preds["label"], "FM embeddings")

    fig.tight_layout()
    fig.savefig(f"figures/flow_matching/{model_name}_{sigma_name}_predictions.png", dpi=300)
    plt.show()
    plt.close()

    fig = umap_compare(vf_preds_embeddings[train_nb:], embeddings[train_nb:], {
        k: ground_truth[k][train_nb:] for k in ground_truth
    })
    fig.savefig(f"figures/flow_matching/{model_name}_{sigma_name}_umap.png", dpi=300)
    plt.show()
    plt.close()