import torch, argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
from umap import UMAP
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score

from scripts.plots.plot_utils import plot_cross_section_histogram, set_fonts
from scripts.embedings_utils import merge_datasets
from scripts.model_utils import LinearRegression, compute_embeddings, load_astropt_model, load_fm_model

def get_list_conditions(condition_names):

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


def plot_results(fm_model, train_embeddings, val_embeddings, val_cond, label_names):
    """
    Generate all plots for the training results.
    
    Args:
        fm_model: Trained VectorField model
        train_embeddings: Training embedding tensor
        val_embeddings: Validation embedding tensor
        val_cond: Validation conditions dictionary
        train_avg_meter: Training loss meter
        val_avg_meter: Validation loss meter
        args: Argument parser with configuration
    
    Returns:
        figures: List of (figure, name, filename) tuples
        metrics: Dictionary of computed metrics
    """
    # Make predictions for plots
    lin_reg = LinearRegression(train_embeddings.device).fit(train_embeddings, torch.stack([val_cond[k] for k in fm_model.config.conditions], dim=-1))
    lin_preds_array = lin_reg.predict(val_embeddings).cpu().numpy()

    vf_embeddings = fm_model.sample_flow(torch.stack([val_cond[k] for k in fm_model.config.conditions], dim=-1))
    vf_preds_array = lin_reg.predict(vf_embeddings).cpu().numpy()

    lin_preds = {label_name: lin_preds_array[:, i] for i, label_name in enumerate(fm_model.config.conditions)}
    vf_preds = {cond_name: vf_preds_array[:, i] for i, cond_name in enumerate(fm_model.config.conditions)}
    val_cond_dict = {label_name: val_cond[label_name].cpu().numpy() if hasattr(val_cond[label_name], 'cpu') else val_cond[label_name] 
                     for label_name in fm_model.config.conditions}

    figures = []
    metrics = {}

    # Plot predictions (except for cross sections)
    for cond_name in filter(lambda x: "label" not in x, label_names):
        fig, (lin_ax, fm_ax) = plt.subplots(1, 2, figsize=(12, 6))

        lin_reg_cond = LinearRegression("cpu").fit(val_cond_dict[cond_name], lin_preds[cond_name])
        slope = lin_reg_cond.weights.item()
        intercept = lin_reg_cond.bias.item()

        mse = mean_squared_error(val_cond_dict[cond_name], lin_preds[cond_name])
        r2 = r2_score(val_cond_dict[cond_name], lin_preds[cond_name])

        lin_ax.scatter(val_cond_dict[cond_name], lin_preds[cond_name], alpha=0.3)
        lin_ax.plot(val_cond_dict[cond_name], val_cond_dict[cond_name] * slope + intercept, 
                    label=f"y={slope:.2f}x + {intercept:.2f}", color='red')
        
        lin_ax.set_title(f"Val embeddings predictions for {cond_name}\nMSE: {mse:.4f}, R2: {r2:.4f}")
        lin_ax.set_xlabel(f"Ground truth {cond_name}")
        lin_ax.set_ylabel(f"Predicted {cond_name}")
        lin_ax.legend()

        rel_diff = np.abs(lin_preds[cond_name] - val_cond_dict[cond_name]) / np.maximum(np.abs(val_cond_dict[cond_name]), 1e-6)
        mse = mean_squared_error(val_cond_dict[cond_name], vf_preds[cond_name])
        r2 = r2_score(val_cond_dict[cond_name], vf_preds[cond_name])

        ax_col = fm_ax.scatter(val_cond_dict[cond_name], vf_preds[cond_name], c=rel_diff, alpha=0.3)
        cbar = fig.colorbar(ax_col, ax=fm_ax, label='Relative difference')
        fm_ax.plot(val_cond_dict[cond_name], val_cond_dict[cond_name] * slope + intercept, color='red')

        fm_ax.set_title(f"Predictions with FM embeddings for {cond_name} \nMSE: {mse:.4f}, R2: {r2:.4f}")
        fm_ax.set_xlabel(f"Ground truth {cond_name}")
        fm_ax.set_ylabel(f"Predicted {cond_name}")

        fig.tight_layout()

        metrics[f"{cond_name}_mse"] = mse
        metrics[f"{cond_name}_r2"] = r2

        figures.append((fig, f"{cond_name} Predictions", f"{cond_name}_predictions.png"))
    
    # Plot cross-section for label if available
    if "label" in label_names or "log_label" in label_names:
        fig, (lin_ax, fm_ax) = plt.subplots(1, 2, figsize=(12, 6))
        key = "label" if "label" in label_names else "log_label"

        min_x = min(lin_preds[key].min(), vf_preds[key].min())
        max_x = max(lin_preds[key].max(), vf_preds[key].max())

        plot_cross_section_histogram(lin_ax,
            val_cond_dict[key], lin_preds[key], 
            bin_range=(min_x, max_x),
            pred_method_name="Linear Regression")
        
        plot_cross_section_histogram(fm_ax,
            val_cond_dict[key], vf_preds[key], 
            bin_range=(min_x, max_x),
            pred_method_name="Flow Matching + Linear Regression")
        
        mse = mean_squared_error(val_cond_dict[key], vf_preds[key])
        r2 = r2_score(val_cond_dict[key], vf_preds[key])

        metrics[key + "_mse"] = mse
        metrics[key + "_r2"] = r2

        figures.append((fig, f"{key} Predictions", f"{key}_predictions.png"))

    # Plot UMAP projections of embeddings
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
    figures.append((fig, "UMAP Embeddings", "umap_embeddings.png"))

    return figures, metrics


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