"""
Trains a flow matching model on astroPT embeddings.

python3 -m scripts.train_generator \
    --model_path <astropt_model_path> \
    --nb_points 14000 --epochs 5000 --sigma 1.0
"""
import numpy as np
import argparse, wandb, os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from umap import UMAP
from dataclasses import asdict
from sklearn.metrics import mean_squared_error, r2_score

from flow_model.vector_field import VectorField, VectorFieldConfig
from scripts.model_utils import RunningAverageMeter, LinearRegression, load_astropt_model, load_fm_model
from scripts.plot_utils import plot_cross_section_histogram
from scripts.embedings_utils import merge_datasets, compute_embeddings


def get_datasets(model_path, label_names, split_ratio=0.8, nb_points=14000):
    model = load_astropt_model(model_path, device=device, strict=True)
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"],
        feature_names=label_names, stack_features=False) \
            .shuffle(seed=42) \
            .take(nb_points)    

    has_metals = device.type == 'mps'
    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 256,
        num_workers = 0 if has_metals else 4,
        prefetch_factor = None if has_metals else 3
    )

    embeddings, cond_dict = compute_embeddings(model, dl, device, label_names)
    cond = torch.stack([cond_dict[k] for k in label_names], dim=-1)

    # Split into train and val
    nb_train = int(split_ratio * embeddings.size(0))

    train_embeddings = embeddings[:nb_train]
    val_embeddings = embeddings[nb_train:]
    train_cond = cond[:nb_train]
    val_cond = cond[nb_train:]

    return (train_embeddings, val_embeddings), (train_cond, val_cond)

def compute_loss(fm_model: VectorField, x0, x1, cond):
    t, xt, ut = fm_model.solver.sample_location_and_conditional_flow(x0, x1)
    return F.mse_loss(fm_model(t, xt, cond), ut)

def sample_batch(data_loader, device):
    for x1, cond in data_loader:
        x1 = x1.to(device)
        cond = cond.to(device)
        x0 = torch.randn_like(x1, device=device)  

        yield x0, x1, cond

def train_model(train_dl: DataLoader, val_dl: DataLoader, fm_model: VectorField, 
                wandb_run, epochs, checkpoint_path) -> VectorField:
    
    train_avg_meter = RunningAverageMeter() 
    val_avg_meter = RunningAverageMeter() 
    opt = torch.optim.AdamW(fm_model.parameters(), lr=1e-3)

    best_model = None
    best_val_loss = float('inf')
    for epoch in range(epochs):

        fm_model.train()
        train_loss = 0.0
        for x0, x1, cond in sample_batch(train_dl, device):

            loss = compute_loss(fm_model, x0, x1, cond)

            opt.zero_grad()
            loss.backward()
            opt.step()
                
            train_loss += loss.item()

        train_loss /= len(train_dl)

        fm_model.eval()
        val_loss = 0.0
        for x0, x1, cond in sample_batch(val_dl, device):

            loss = compute_loss(fm_model, x0, x1, cond)

            val_loss += loss.item()

        val_loss /= len(val_dl)

        train_avg_meter.update(train_loss)
        val_avg_meter.update(val_loss)
        
        if epoch % 10 == 0:
            if wandb_run is not None:
                wandb_run.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
            else:
                print(f"Epoch {epoch}: train_loss: {train_avg_meter.avg:.9f}, val_loss: {val_avg_meter.avg:.9f}")

            if val_avg_meter.avg < best_val_loss:
                best_model = fm_model.state_dict()
                best_val_loss = val_avg_meter.avg
                torch.save({
                        "state_dict": best_model,
                        "config": asdict(fm_model.config),
                        "conditions": fm_model.config.conditions,
                        "val_loss": best_val_loss
                    }, checkpoint_path)
                
    print(f"Training completed. Best val loss: {best_val_loss:.4f}\n\n")
    if best_model is None:
        return fm_model
    
    fm_model.load_state_dict(best_model)
    return fm_model


def main(args, device):
    print(f"Generating embeddings on device: {device}")

    model_name = args.model_path.split('/')[-1].replace('.pt','')
    sigma_name = f"sigma_{args.sigma:.3f}".replace('.','_')
    path_prefix = args.path_prefix + '_' if args.path_prefix else ""
    checkpoint_path = f"model/flow_matching/{path_prefix}{args.ot_method}_{sigma_name}_{model_name}.pt"

    # ==============================================
    # Setup Wandb
    # ==============================================

    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            entity="matttsss-epfl",
            project="Embedding Generation FM",
            name=f"OT Method: {args.ot_method}, Sigma: {args.sigma:.3f}",
            config={
                "epochs": args.epochs,
                "nb_points": args.nb_points,
                "conditions": args.labels,
                "sigma": args.sigma,
                "ot_method": args.ot_method,
                "astropt_model": model_name
            }
        )
    
    # Compute datasets
    (train_embeddings, val_embeddings), (train_cond, val_cond) = \
        get_datasets(args.model_path, args.labels, split_ratio=0.8, nb_points=args.nb_points)

    nb_train = train_embeddings.size(0)
    nb_embeddings = train_embeddings.size(0) + val_embeddings.size(0)
    embeddings_dim = train_embeddings.size(1)
    print(f"Train embeddings: {train_embeddings.shape}, Val embeddings: {val_embeddings.shape}")

    # ==============================================
    # Train Flow Matching model
    # ==============================================

    train_embed_dl = DataLoader(TensorDataset(train_embeddings, train_cond), 
                                batch_size=nb_train)
    val_embed_dl = DataLoader(TensorDataset(val_embeddings, val_cond), 
                              batch_size=nb_embeddings - nb_train)
    
    fm_config = VectorFieldConfig(
        sigma=args.sigma,
        dim=embeddings_dim,
        ot_method=args.ot_method,
        conditions=args.labels
    )
    fm_model = VectorField(fm_config, num_threads=4).to(device)
    
    try:
        fm_model = train_model(train_embed_dl, val_embed_dl, fm_model, wandb_run, args.epochs, checkpoint_path)
    except KeyboardInterrupt:
        print("Training interrupted. Proceeding to validation with current model.")
        fm_model = load_fm_model(checkpoint_path, device=device, strict=False)
    # ==============================================
    # Make predictions for plots
    # ==============================================

    lin_reg = LinearRegression(device).fit(train_embeddings, train_cond)
    lin_preds = lin_reg.predict(val_embeddings).cpu().numpy()

    vf_embeddings = fm_model.sample_flow(val_cond)
    vf_preds = lin_reg.predict(vf_embeddings).cpu().numpy()

    lin_preds = {label_name: lin_preds[:, i] for i, label_name in enumerate(fm_model.config.conditions)}
    vf_preds = {cond_name: vf_preds[:, i] for i, cond_name in enumerate(fm_model.config.conditions)}
    val_cond = {label_name: val_cond[:, i].cpu().numpy() for i, label_name in enumerate(fm_model.config.conditions)}


    # ==============================================
    # Plot predictions (except for cross sections)
    # ==============================================

    plot_folder = f"figures/flow_matching/{args.ot_method}/{path_prefix}{sigma_name}_{model_name}"
    if args.save_plots or wandb_run is None: os.makedirs(plot_folder, exist_ok=True)

    metrics = {}
    
    for cond_name in filter(lambda x: "label" not in x, args.labels):
        fig, (lin_ax, fm_ax) = plt.subplots(1, 2, figsize=(12, 6))

        lin_reg = LinearRegression("cpu").fit(val_cond[cond_name], lin_preds[cond_name])
        slope = lin_reg.weights.item()
        intercept = lin_reg.bias.item()

        mse = mean_squared_error(val_cond[cond_name], lin_preds[cond_name])
        r2 = r2_score(val_cond[cond_name], lin_preds[cond_name])

        lin_ax.scatter(val_cond[cond_name], lin_preds[cond_name], alpha=0.3)
        lin_ax.plot(val_cond[cond_name], val_cond[cond_name] * slope + intercept, 
                    label=f"y={slope:.2f}x + {intercept:.2f}", color='red')
        
        lin_ax.set_title(f"Val embeddings predictions for {cond_name}\nMSE: {mse:.4f}, R2: {r2:.4f}")
        lin_ax.set_xlabel(f"Ground truth {cond_name}")
        lin_ax.set_ylabel(f"Predicted {cond_name}")
        lin_ax.legend()

        rel_diff = np.abs(lin_preds[cond_name] - val_cond[cond_name]) / np.maximum(np.abs(val_cond[cond_name]), 1e-6)
        mse = mean_squared_error(val_cond[cond_name], vf_preds[cond_name])
        r2 = r2_score(val_cond[cond_name], vf_preds[cond_name])

        ax_col = fm_ax.scatter(val_cond[cond_name], vf_preds[cond_name], c=rel_diff, alpha=0.3)
        cbar = fig.colorbar(ax_col, ax=fm_ax, label='Relative difference')
        fm_ax.plot(val_cond[cond_name], val_cond[cond_name] * slope + intercept, color='red')

        fm_ax.set_title(f"Predictions with FM embeddings for {cond_name} \nMSE: {mse:.4f}, R2: {r2:.4f}")
        fm_ax.set_xlabel(f"Ground truth {cond_name}")
        fm_ax.set_ylabel(f"Predicted {cond_name}")

        fig.tight_layout()

        metrics[f"{cond_name}_mse"] = mse
        metrics[f"{cond_name}_r2"] = r2

        if wandb_run is not None:
            wandb_run.log({f"{cond_name}_predictions": wandb.Image(fig)})
        if args.save_plots or wandb_run is None:
            fig.savefig(f"{plot_folder}/{cond_name}_predictions.png", dpi=300)
            print(f"Saved plot for {cond_name} predictions.")
        
        plt.close(fig)
    
    # ==============================================
    # Plot cross-section for label if available
    # ==============================================

    if "label" in args.labels or "log_label" in args.labels:
        fig, (lin_ax, fm_ax) = plt.subplots(1, 2, figsize=(12, 6))
        key = "label" if "label" in args.labels else "log_label"

        min_x = min(lin_preds[key].min(), vf_preds[key].min())
        max_x = max(lin_preds[key].max(), vf_preds[key].max())

        plot_cross_section_histogram(lin_ax,
            val_cond[key], lin_preds[key], 
            bin_range=(min_x, max_x),
            pred_method_name="Linear Regression")
        
        plot_cross_section_histogram(fm_ax,
            val_cond[key], vf_preds[key], 
            bin_range=(min_x, max_x),
            pred_method_name="Flow Matching + Linear Regression")
        
        if wandb_run is not None:
            wandb_run.log({f"label_predictions": wandb.Image(fig)})
        if args.save_plots or wandb_run is None:
            fig.savefig(f"{plot_folder}/{key}_predictions.png", dpi=300)
            print(f"Saved plot for {key} predictions.")
        
        plt.close(fig)

        mse = mean_squared_error(val_cond[key], vf_preds[key])
        r2 = r2_score(val_cond[key], vf_preds[key])

        metrics[key + "_mse"] = mse
        metrics[key + "_r2"] = r2

    # ==============================================
    # Plot UMAP projections of embeddings
    # ==============================================

    umap = UMAP(n_components=2).fit(train_embeddings.cpu().numpy())
    val_umap = umap.transform(val_embeddings.cpu().numpy())
    vf_umap_embeddings = umap.transform(vf_embeddings.cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(val_umap[:, 0], val_umap[:, 1], alpha=0.3)
    axs[0].set_title("UMAP of Validation Embeddings")

    diff_umap = np.linalg.norm(val_umap - vf_umap_embeddings, axis=-1)
    ax_col = axs[1].scatter(vf_umap_embeddings[:, 0], vf_umap_embeddings[:, 1], c=diff_umap, alpha=0.3)
    cbar = fig.colorbar(ax_col, ax=axs[1], label='L2 difference')
    axs[1].set_title("UMAP of Flow Matching Embeddings")

    fig.tight_layout()
    if wandb_run is not None:
        wandb_run.log({f"umap_embeddings": wandb.Image(fig)})
    if args.save_plots or wandb_run is None:
        fig.savefig(f"{plot_folder}/umap_embeddings.png", dpi=300)
        print("Saved plot for UMAP embeddings.")

    plt.close(fig)

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    checkpoint["metrics"] = metrics
    torch.save(checkpoint, checkpoint_path)
    print("Saved metrics to checkpoint.")

    if wandb_run is not None:
        for metric_name, metric_vals in metrics.items():
            wandb_run.summary[metric_name] = metric_vals
        wandb_run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='GenAstroPT',
                    description='Trains a flow matching model on astroPT embeddings')
    parser.add_argument('--nb_points', type=int, default=7000, help='Number of points to use for embeddings')
    parser.add_argument('--labels', nargs='+', default=["mass", "label"], help='Labels to use for the conditions of the flow matching model')
    parser.add_argument('--model_path', type=str, default="model/ckpt.pt", help='Path to the astropt checkpoint')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma value for the flow matching model')
    parser.add_argument('--ot_method', type=str, default="default", help='Optimal transport method to use')

    parser.add_argument('--path_prefix', type=str, default="", help='Path prefix for saving the model and plots')
    parser.add_argument('--save_plots', action='store_true', help='Whether to save plots locally')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb for logging')
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    
    main(args, device)
