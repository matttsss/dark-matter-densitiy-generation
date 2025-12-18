"""
Flow Matching Model Training Module

This module trains a Flow Matching model on AstroPT embeddings using conditional flows
to learn the mapping between conditions (e.g., mass, redshift) and embedding space.
Supports Optimal Transport-based conditioning and provides training monitoring via W&B.

Example:
    To train a flow matching model on AstroPT embeddings with custom conditions:
    
    $ python3 -m scripts.flow_model_train \\
        --model_path model_weights/best_r_ell_model.pt \\
        --nb_points 14000 \\
        --epochs 5000 \\
        --sigma 1.0 \\
        --labels mass label \\
        --save_plots \\
        --use_wandb
    
    Arguments:
        --model_path: Path to trained AstroPT embedding model
        --nb_points: Number of embeddings to train on (default: 7000)
        --epochs: Number of training epochs (default: 3000)
        --batch_scale: Batch scaling factor for time sampling (default: 20)
        --sigma: Flow matching sigma parameter (default: 0.1)
        --ot_method: Optimal transport method to use (default: "default")
        --encoding_size: Hash grid encoding size (default: 64)
        --mlp_hidden_dim: MLP hidden layer dimension (default: 1024)
        --lr: Learning rate (default: 5e-4)
        --labels: Space-separated condition names
        --path_prefix: Prefix for model and plot paths
        --save_plots: Save plots locally
        --use_wandb: Log metrics to Weights & Biases
        --checkpoint: Path to checkpoint for resuming training
"""
import numpy as np
import argparse, wandb, os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from dataclasses import asdict

from generative_model.vector_field import VectorField, VectorFieldConfig
from scripts.plots.plot_utils import set_fonts
from scripts.model_utils import RunningAverageMeter, load_fm_model, get_embeddings_datasets
from scripts.plots.fm_validation import plot_results


def compute_loss(fm_model: VectorField, x0, x1, cond, multiplicator: int = 1):
    """
    Compute flow matching loss between initial and target distributions.
    
    Samples random time points and intermediate positions along the flow path,
    then computes MSE loss between predicted and target velocity fields.
    
    Args:
        fm_model (VectorField): Flow matching model
        x0 (torch.Tensor): Initial samples from noise distribution, shape (batch_size, embedding_dim)
        x1 (torch.Tensor): Target samples from data distribution, shape (batch_size, embedding_dim)
        cond (torch.Tensor): Condition tensors, shape (batch_size, num_conditions)
        multiplicator (int): Factor to repeat samples for multiple time samples per data point (default: 1)
    
    Returns:
        torch.Tensor: Scalar MSE loss value
    """
    if multiplicator > 1:
        x0 = x0.repeat(multiplicator, 1)
        x1 = x1.repeat(multiplicator, 1)
        cond = cond.repeat(multiplicator, 1)

    t = torch.rand(x0.size(0), device=x0.device)
    t, xt, ut = fm_model.solver.sample_location_and_conditional_flow(x0, x1, t)
    return F.mse_loss(fm_model(t, xt, cond), ut)

def sample_batch(data_loader, device):
    """
    Generator that yields batches with noise samples paired with data samples.
    
    For each batch from the data loader, generates random noise samples (x0) and
    pairs them with actual data embeddings (x1) and conditions.
    
    Args:
        data_loader (DataLoader): Data loader yielding (embeddings, conditions) tuples
        device (torch.device): Device for tensor allocation
    
    Yields:
        tuple: (x0, x1, cond) where x0 is noise, x1 is data, cond is conditions
    """
    for x1, cond in data_loader:
        x1 = x1.to(device)
        cond = cond.to(device)
        x0 = torch.randn_like(x1, device=device)  

        yield x0, x1, cond

def train_model(train_dl: DataLoader, val_dl: DataLoader, fm_model: VectorField, 
                wandb_run: wandb.Run | None, epochs: int,
                batch_scale: int, checkpoint_path, lr: float = 5e-4) -> VectorField:
    """
    Train flow matching model with validation and checkpointing.
    
    Performs training loop with cosine annealing learning rate scheduling, periodic
    validation, and saves best model based on validation loss. Supports both local
    logging and Weights & Biases integration.
    
    Args:
        train_dl (DataLoader): Training data loader
        val_dl (DataLoader): Validation data loader
        fm_model (VectorField): Flow matching model to train
        wandb_run (wandb.Run | None): W&B run for logging, or None for local logging
        epochs (int): Number of training epochs
        batch_scale (int): Factor to scale batches for increased time sampling
        checkpoint_path (str): Path to save best model checkpoint
        lr (float): Learning rate for AdamW optimizer (default: 5e-4)
    
    Returns:
        VectorField: Trained model (loaded from best checkpoint)
    """
    opt = torch.optim.AdamW(fm_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)

    global train_avg_meter, val_avg_meter

    best_model = None
    best_val_loss = float('inf')
    for epoch in range(epochs):

        fm_model.train()
        train_loss = 0.0
        for x0, x1, cond in sample_batch(train_dl, device):

            loss = compute_loss(fm_model, x0, x1, cond, batch_scale)

            opt.zero_grad()
            loss.backward()
            opt.step()
                
            train_loss += loss.item()

        train_loss /= len(train_dl)

        fm_model.eval()
        val_loss = 0.0
        for x0, x1, cond in sample_batch(val_dl, device):

            loss = compute_loss(fm_model, x0, x1, cond, batch_scale)

            val_loss += loss.item()

        val_loss /= len(val_dl)

        train_avg_meter.update(train_loss)
        val_avg_meter.update(val_loss)
        scheduler.step()
        
        if epoch % 10 == 0:
            if wandb_run is not None:
                wandb_run.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            else:
                print(f"Epoch {epoch}: train_loss: {train_avg_meter.avg:.9f}, val_loss: {val_avg_meter.avg:.9f}")

            train_avg_meter.register_loss(train_loss)
            val_avg_meter.register_loss(val_loss)
            
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
    checkpoint_path = f"model/flow_matching/{path_prefix}{args.ot_method}_{args.mlp_hidden_dim}_{sigma_name}_{model_name}.pt"

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
                "encoding_size": args.encoding_size,
                "mlp_hidden_dim": args.mlp_hidden_dim,
                "astropt_model": model_name
            }
        )
    
    # Compute datasets
    (train_embeddings, val_embeddings), (train_cond, val_cond) = \
        get_embeddings_datasets(args.model_path, device, args.labels, split_ratio=0.8, nb_points=args.nb_points)

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
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading model from checkpoint: {args.checkpoint}")
        fm_model = load_fm_model(args.checkpoint, device=device, strict=False)
    else:
        fm_config = VectorFieldConfig(
            sigma=args.sigma,
            dim=embeddings_dim,
            encoding_size=args.encoding_size,
            ot_method=args.ot_method,
            conditions=args.labels,
            hidden=args.mlp_hidden_dim
        )
        fm_model = VectorField(fm_config, num_threads=4).to(device)
    
    # Global to bypass interupts
    global train_avg_meter, val_avg_meter
    train_avg_meter = RunningAverageMeter(keep_all=True) 
    val_avg_meter = RunningAverageMeter(keep_all=True) 

    try:
        fm_model = train_model(train_embed_dl, val_embed_dl, fm_model, wandb_run,
                               args.epochs, args.batch_scale, checkpoint_path, args.lr)
    except KeyboardInterrupt:
        print("Training interrupted. Proceeding to validation with current model.")
        fm_model = load_fm_model(checkpoint_path, device=device, strict=False)

    # ==============================================
    # Generate plots and metrics
    # ==============================================

    if wandb_run is None:
        cutoff = 15
        fig, ax = plt.subplots()
        ax.plot(train_avg_meter.losses[cutoff:], label='Train Loss')
        ax.plot(val_avg_meter.losses[cutoff:], label='Val Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss over Epochs')
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    
    model_name = args.model_path.split('/')[-1].replace('.pt','')
    sigma_name = f"sigma_{args.sigma:.3f}".replace('.','_')
    path_prefix = args.path_prefix + '_' if args.path_prefix else ""
    plot_folder = f"figures/flow_matching/{args.ot_method}_{args.mlp_hidden_dim}/{path_prefix}{sigma_name}_{model_name}"
    
    figures, metrics = plot_results(fm_model, train_embeddings, train_cond, val_embeddings, val_cond, args.labels)
    
    # Handle figure logging and saving
    if args.save_plots or wandb_run is None:
        os.makedirs(plot_folder, exist_ok=True)
    
    for fig, fig_name, fig_filename in figures:
        if wandb_run is not None:
            wandb_run.log({fig_name: wandb.Image(fig)})
        if args.save_plots or wandb_run is None:
            fig_path = f"{plot_folder}/{fig_filename}"
            fig.savefig(fig_path, dpi=300)
            print(f"Saved plot: {fig_path}")
        
        plt.close(fig)
    
    # Save metrics to checkpoint
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
    parser.add_argument('--model_path', type=str, default="model_weights/baseline_astropt.pt", help='Path to the astropt checkpoint')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--batch_scale', type=int, default=1, help='Batch scales the batches to get more time samples per batch')
    
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma value for the flow matching model')
    parser.add_argument('--ot_method', type=str, default="default", help='Optimal transport method to use')
    parser.add_argument('--encoding_size', type=int, default=64, help='Size of the hash grid encoding')
    parser.add_argument('--mlp_hidden_dim', type=int, default=1024, help='Width of the MLP hidden layers')

    parser.add_argument('--path_prefix', type=str, default="", help='Path prefix for saving the model and plots')
    parser.add_argument('--save_plots', action='store_true', help='Whether to save plots locally')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb for logging')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load model from')
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    
    # Set fonts for plots
    set_fonts()
    main(args, device)
