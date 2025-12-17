"""
Flow Matching Model Inference and Visualization Module

This module performs inference with a trained Flow Matching model by sampling embeddings
for user-specified conditions and comparing them with actual validation embeddings.
It generates visualizations including 2D prediction distributions, UMAP projections,
and per-condition statistics.

Example:
    To run this script as a module to generate embeddings for custom conditions:
    
    $ python3 -m scripts.plots.fm_inference \\
        --fm_model_path model/flow_matching/best_model.pt \\
        --astropt_model_path model/best_r_ell_model.pt \\
        --nb_points 14000 \\
        --nb_gen_points 6000 \\
        --labels mass label

"""

import torch, argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
from umap import UMAP
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

from scripts.plots.plot_utils import set_fonts
from scripts.embeddings_utils import merge_datasets
from scripts.model_utils import LinearRegression, compute_embeddings, load_astropt_model, load_fm_model

def get_list_conditions(condition_names):
    """
    Interactively collect condition values from user input.
    
    Prompts the user to enter values for each condition parameter and collects them
    into a list until the user signals completion (Ctrl+C or EOF).
    
    Args:
        condition_names (list): Names of conditions to prompt for
    
    Returns:
        torch.Tensor: Tensor of shape (num_inputs, len(condition_names)) containing
                      user-specified condition values
    """
    condition_list = []

    try:
        while True:
            print("\nPlease enter the following condition values (Ctrl + C to finish):")
            point = [0 ] * len(condition_names)
            for i, name in enumerate(condition_names):
                value = float(input(f"  {name}: "))
                point[i] = value
            condition_list.append(point)

    except (KeyboardInterrupt, EOFError):
        print("\nInput finished.")

    return torch.tensor(condition_list)

def main(args, device):
    # ================= Load Models =================
    fm_model = load_fm_model(args.fm_model_path, device=device, strict=True)
    fm_model.eval()

    astropt_model = load_astropt_model(args.astropt_model_path, device=device, strict=True)
    astropt_model.eval()

    # =============== Load datasets ===================
    merged_labels = fm_model.config.conditions + [label_name for label_name in args.labels if label_name not in fm_model.config.conditions]
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"],
            feature_names=merged_labels, stack_features=False) \
            .shuffle(seed=42) \
            .take(args.nb_points)
    label_names = [
        (r"$\sigma_{DM}/m$" if "label" == label_name else label_name) for label_name in merged_labels
    ]

    dl = DataLoader(
        dataset,
        batch_size = 64 if (device.type == "mps") else 128,
        num_workers = 0 if (device.type == "mps") else 4,
        prefetch_factor = None if (device.type == "mps") else 3
    )

    # ============== Compute embeddings =================
    embeddings, labels = compute_embeddings(astropt_model, dl, device, merged_labels)
    del dataset, dl, astropt_model
    torch.cuda.empty_cache()

    lin_reg = LinearRegression(device).fit(embeddings, 
                torch.stack([labels[label_name] for label_name in fm_model.config.conditions], dim=1))

    conditions = get_list_conditions(fm_model.config.conditions)
    if conditions.ndim == 1:
        conditions = conditions.unsqueeze(0)
    conditions = conditions.to(device)

    expanded_conditions = conditions.repeat_interleave(args.nb_gen_points, dim=0)

    # ============== Predict and plot =================
    fm_embeddings = fm_model.sample_flow(expanded_conditions)
    preds = lin_reg.predict(fm_embeddings).cpu().numpy()
    expanded_conditions_np = expanded_conditions.cpu().numpy()
    base_conditions_np = conditions.cpu().numpy()

    file_suffix = f"batch_{len(base_conditions_np)}_conds"

    mse = np.mean((preds - expanded_conditions_np)**2)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get axis limits based on labels min/max
    label_vals_0 = labels[fm_model.config.conditions[0]].cpu().numpy()
    label_vals_1 = labels[fm_model.config.conditions[1]].cpu().numpy()
    x_min, x_max = label_vals_0.min(), label_vals_0.max()
    y_min, y_max = label_vals_1.min(), label_vals_1.max()
    
    # Create single histogram combining all generated embeddings
    ax_colors = ax.hist2d(preds[:, 0], preds[:, 1], bins=60, cmap="viridis",
                          range=[[x_min, x_max], [y_min, y_max]])
    
    # Overlay per-condition centers and covariance ellipses
    for idx in range(base_conditions_np.shape[0]):
        start = idx * args.nb_gen_points
        end = start + args.nb_gen_points
        cond_preds = preds[start:end]
        mu = cond_preds.mean(axis=0)
        cov = np.cov(cond_preds.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        cond_truth = expanded_conditions_np[start:end]

        # 2-sigma ellipse
        width, height = 4 * np.sqrt(vals)
        ell = Ellipse(xy=mu, width=width, height=height, angle=theta,
                      edgecolor='red', facecolor='none', linewidth=1.5, alpha=0.8)
        
        ax.add_patch(ell)
        ax.scatter(mu[0], mu[1], color='red', s=40, zorder=6)
        
        # Per-axis MSE labels near ellipse center
        mse_x = mean_squared_error(cond_preds[:, 0], cond_truth[:, 0])
        mse_y = mean_squared_error(cond_preds[:, 1], cond_truth[:, 1])
        label_text = f"MSE x: {mse_x:.4f}\nMSE y: {mse_y:.4f}"
        ax.text(mu[0], mu[1], label_text,
            color='white', ha='left', va='bottom',
            bbox=dict(facecolor='black', alpha=0.5, pad=3))

    # Also show true condition points
    ax.scatter(base_conditions_np[:, 0], base_conditions_np[:, 1],
               color='red', label='True Values', marker='x', s=100, zorder=5)
    
    cbar = fig.colorbar(ax_colors[3], ax=ax, label="Density")
    cbar.set_label("Embeddings per prediction bin")
    ax.set_xlabel(f"{label_names[0]}")
    ax.set_ylabel(f"{label_names[1]}")
    ax.set_title("Flow Matching Predictions vs True Values\nMSE: {:.4f}".format(mse))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figures/flow_matching/inference/fm_gen_{file_suffix}.png")
    print(f"Saved figure to figures/flow_matching/inference/fm_gen_{file_suffix}.png")
    plt.show()
    plt.close(fig)

    # ============ UMAP embeddings plot =================
    umap_model = UMAP(n_components=2)
    embeddings_2d = umap_model.fit_transform(embeddings.cpu().numpy())
    fm_umap_embeddings = umap_model.transform(fm_embeddings.cpu().numpy())
    
    # Calculate shared axis limits
    x_min = min(embeddings_2d[:, 0].min(), fm_umap_embeddings[:, 0].min())
    x_max = max(embeddings_2d[:, 0].max(), fm_umap_embeddings[:, 0].max())
    y_min = min(embeddings_2d[:, 1].min(), fm_umap_embeddings[:, 1].min())
    y_max = max(embeddings_2d[:, 1].max(), fm_umap_embeddings[:, 1].max())

    # AstroPT + per-condition Flow Matching UMAP histograms in one figure
    n_conditions = base_conditions_np.shape[0]
    n_cols = n_conditions + 1
    fig, axs = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), sharex=True, sharey=True)

    # AstroPT reference
    axs[0].hist2d(embeddings_2d[:, 0], embeddings_2d[:, 1], bins=60,
                  range=[[x_min, x_max], [y_min, y_max]], cmap="viridis")
    axs[0].set_title("UMAP AstroPT")
    axs[0].set_xlabel("UMAP 1")
    axs[0].set_ylabel("UMAP 2")

    # One Flow Matching UMAP plot per condition batch
    for idx in range(n_conditions):
        start = idx * args.nb_gen_points
        end = start + args.nb_gen_points
        cond_umap = fm_umap_embeddings[start:end]

        ax = axs[idx + 1]
        ax.hist2d(cond_umap[:, 0], cond_umap[:, 1], bins=60,
                  range=[[x_min, x_max], [y_min, y_max]], cmap="viridis", label=f"Cond {idx+1}")
        plot_title = ", ".join([f"{name}={base_conditions_np[idx, i]:.2f}" 
                                for i, name in enumerate(label_names)])
        ax.set_title(f"UMAP Flow Matching Embeddings\n{plot_title}")
        ax.set_xlabel("UMAP 1")


    fig.tight_layout()
    fig.savefig(f"figures/flow_matching/inference/fm_umap_{file_suffix}.png")
    print(f"Saved figure to figures/flow_matching/inference/fm_umap_{file_suffix}.png")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Validate Flow Matching Model")
    argparser.add_argument("--fm_model_path", type=str, required=True,
                            help="Path to the Flow Matching model to validate")
    argparser.add_argument("--astropt_model_path", type=str, required=True,
                            help="Path to the AstroPT model to use as reference")
    argparser.add_argument("--nb_points", type=int, default=14000,
                            help="Number of points to use for validation")
    argparser.add_argument("--nb_gen_points", type=int, default=6000,
                            help="Number of points to generate with the Flow Matching model")
    argparser.add_argument("--labels", nargs='+', default=["mass", "label"], help="Physical quantities to plot")
    args = argparser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")

    set_fonts()
    main(args, device)