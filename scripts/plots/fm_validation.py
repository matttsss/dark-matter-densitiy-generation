"""
Flow Matching Model Validation and Visualization Module

This module generates validation plots for a Flow Matching model by comparing
predictions made with linear regression on original embeddings versus embeddings sampled
from the flow matching model. It provides visualization of prediction accuracy, embedding
distributions, and UMAP projections.

Example:
    To run this script as a module with a Flow Matching model and AstroPT embeddings,
    generating predictions for 'mass' and 'label' conditions:
    
    $ python3 -m scripts.plots.fm_validation \\
        --fm_model_path model/flow_matching/best_model.pt \\
        --astropt_model_path model/best_r_ell_model.pt \\
        --labels mass label \\
        --nb_points 8000 \\
        --save_plots

"""

import torch, argparse
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.metrics import mean_squared_error, r2_score

from scripts.plots.plot_utils import plot_cross_section_histogram, set_fonts
from scripts.model_utils import LinearRegression, load_fm_model, get_embeddings_datasets

def plot_results(fm_model, train_embeddings, train_cond, val_embeddings, val_cond, label_names):
    """
    Generate validation plots comparing generated embeddings with the original AstroPT embeddings.

    Args:
        fm_model (VectorField): Trained flow matching model for sampling embeddings
        train_embeddings (torch.Tensor): Training set embeddings of shape (N, embedding_dim)
        train_cond (torch.Tensor): Training set conditions of shape (N, num_conditions)
        val_embeddings (torch.Tensor): Validation set embeddings of shape (M, embedding_dim)
        val_cond (torch.Tensor): Validation set conditions of shape (M, num_conditions)
        label_names (list): List of condition names corresponding to columns in train_cond/val_cond
    
    Returns:
        tuple: 
            - figures (list): List of (plt.Figure, title_str, filename_str) tuples
            - metrics (dict): Dictionary mapping condition names to computed MSE and R² scores
    """
    # Make predictions for plots
    lin_reg = LinearRegression(train_embeddings.device).fit(train_embeddings, train_cond)
    lin_preds_array = lin_reg.predict(val_embeddings).cpu().numpy()

    # Sample flow matching embeddings
    vf_embeddings = fm_model.sample_flow(val_cond)
    vf_preds_array = lin_reg.predict(vf_embeddings).cpu().numpy()

    lin_preds = {label_name: lin_preds_array[:, i] for i, label_name in enumerate(fm_model.config.conditions)}
    vf_preds = {cond_name: vf_preds_array[:, i] for i, cond_name in enumerate(fm_model.config.conditions)}
    val_cond_dict = {label_name: val_cond[:, i].cpu().numpy() for i, label_name in enumerate(fm_model.config.conditions)}

    figures = []
    metrics = {}

    # Plot predictions (except for cross sections)
    for cond_name in filter(lambda x: "label" not in x, label_names):
        fig, (lin_ax, fm_ax) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot validation predictions for linear regression
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

        # Plot validation predictions for flow matching embeddings
        lin_reg_cond = LinearRegression("cpu").fit(val_cond_dict[cond_name], lin_preds[cond_name])
        slope = lin_reg_cond.weights.item()
        intercept = lin_reg_cond.bias.item()
        mse = mean_squared_error(val_cond_dict[cond_name], vf_preds[cond_name])
        r2 = r2_score(val_cond_dict[cond_name], vf_preds[cond_name])

        fm_ax.scatter(val_cond_dict[cond_name], vf_preds[cond_name], alpha=0.3)
        fm_ax.plot(val_cond_dict[cond_name], val_cond_dict[cond_name] * slope + intercept, 
                    label=f"y={slope:.2f}x + {intercept:.2f}", color='red')
        fm_ax.legend()

        fm_ax.set_title(f"Predictions with FM embeddings for {cond_name}\nMSE: {mse:.4f}, R2: {r2:.4f}")
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
        fig.tight_layout()

        figures.append((fig, f"{key} Predictions", f"{key}_predictions.png"))

    # Plot UMAP projections of embeddings
    umap = UMAP(n_components=2).fit(train_embeddings.cpu().numpy())
    val_umap = umap.transform(val_embeddings.cpu().numpy())
    vf_umap_embeddings = umap.transform(vf_embeddings.cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(val_umap[:, 0], val_umap[:, 1], alpha=0.3)
    axs[0].set_title("UMAP of Validation Embeddings")

    axs[1].scatter(vf_umap_embeddings[:, 0], vf_umap_embeddings[:, 1], alpha=0.3)
    axs[1].set_title("UMAP of Flow Matching Embeddings")

    fig.tight_layout()
    figures.append((fig, "UMAP Embeddings", "umap_embeddings.png"))

    return figures, metrics

def main(args, device):

    # Load FM model
    fm_model = load_fm_model(args.fm_model_path, device)

    # Compute datasets
    (train_embeddings, val_embeddings), (train_cond, val_cond) = \
        get_embeddings_datasets(args.astropt_model_path, device, args.labels, split_ratio=0.8, nb_points=args.nb_points)

    # Generate plots and metrics
    figures, _ = plot_results(fm_model, train_embeddings, train_cond, val_embeddings, val_cond, args.labels)

    # Save or display plots
    for fig, _, filename in figures:
        if args.save_plots:
            fig.savefig(f"figures/{filename}", dpi=300)
            print(f"Saved figure to figures/{filename}")
        else:
            plt.show()
        plt.close(fig)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Flow Matching Model Validation Plots")
    argparser.add_argument("--fm_model_path", type=str, required=True, help="Path to the trained flow matching model")
    argparser.add_argument("--astropt_model_path", type=str, required=True, help="Path to the trained AstroPT model")
    argparser.add_argument("--labels", type=str, nargs='+', required=True, help="List of label names to validate")
    argparser.add_argument("--nb_points", type=int, default=7000, help="Number of points to use for validation")
    argparser.add_argument("--save_plots", action='store_true', help="Whether to save the plots to files")
    args = argparser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")

    set_fonts()
    main(args, device)
