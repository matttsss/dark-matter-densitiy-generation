"""
Example script that samples from a trained astropt model and finetunes
embeddings on a linear probe
"""

import os
import numpy as np
import torch, argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from scripts.model_utils import compute_embeddings, load_astropt_model
from scripts.plots.plot_utils import plot_labels, set_fonts
from scripts.embeddings_utils import merge_datasets

from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PlotstroPT',
                    description='Generates plots for linear probe on astroPT embeddings')
    
    parser.add_argument('--nb_points', type=int, default=1000, help='Number of points to use for embeddings')
    parser.add_argument('--labels', nargs='+', default=["mass"], help='Labels to use for embeddings')
    parser.add_argument('--model_path', type=str, default="model/ckpt.pt", help='Path to the model checkpoint')
    parser.add_argument('--output_path', type=str, default="figures/finetuned", help='Path to save the output figures')
    args = parser.parse_args()
    
    # Set fonts for plots
    set_fonts()
    
    labels_name = args.labels  

    has_metals = torch.backends.mps.is_available()  
    device = torch.device('mps' if has_metals else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    print(f"Generating embeddings on device: {device}")


    weights_filename = args.model_path
    model = load_astropt_model(weights_filename, device=device, strict=False)
 
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"], 
        labels_name, stack_features=False) \
        .shuffle(seed=42) \
        .take(args.nb_points)    

    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 128,
        num_workers = 0 if has_metals else 4,
        prefetch_factor = None if has_metals else 3
    )

    embeddings, labels = compute_embeddings(model, dl, device, labels_name)
    embeddings = embeddings.cpu().numpy()
    labels = {k: v.cpu().numpy() for k, v in labels.items()}

    data_name = weights_filename.replace(".pt", "").split("/")[-1]
    plot_out_path = args.output_path
    if not os.path.exists(plot_out_path):
        os.makedirs(plot_out_path, exist_ok=True)
    print(f"Plotting probes for {data_name} model...")
    print(f"Saving figures to {plot_out_path}")

    ## Start with UMAP projection
    umap = UMAP(n_components=2)
    umap_embeddings = StandardScaler().fit_transform(embeddings)
    umap_embeddings = umap.fit_transform(umap_embeddings)


    # =============================================================================
    # Plot UMAP projections colored by each label
    # =============================================================================

    def plot_probe(fig, ax, label_name, umap_embeddings, labels_dict):
        data = labels_dict[label_name]
        vmax = np.percentile(data, 95)
        sc = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=data, vmax=vmax, cmap="viridis")

        ax.set_title(label_name)
        fig.colorbar(sc, ax=ax, label=label_name)

    fig = plot_labels(plot_probe, f"UMAP projection of {data_name} embeddings", list(labels.keys()),
                      umap_embeddings=umap_embeddings, labels_dict=labels)

    fig.savefig(f"{plot_out_path}/{data_name}_umap_magnitudes.png", dpi=300)
    plt.show()
    plt.close(fig)


    # =============================================================================
    # Plot Probe predictions
    # =============================================================================
    halfway = len(embeddings) // 2

    def plot_probe_predictions(fig, ax, label_name, embeddings, labels_dict):
        label_data = labels_dict[label_name]

        # Normalize label data
        mean, std = label_data.mean(), label_data.std()
        label_data = (label_data - mean) / std

        # Train linear probe to predict label from embeddings
        probe = LinearRegression().fit(embeddings[:halfway], label_data[:halfway])
        pss = probe.predict(embeddings[halfway:])

        # Compute metrics
        mse = mean_squared_error(pss, label_data[halfway:])
        r2 = r2_score(pss, label_data[halfway:])

        # Plot predictions vs ground truth
        reg = LinearRegression().fit(label_data[halfway:].reshape(-1, 1), pss)
        ax.scatter(label_data[halfway:], pss, alpha=0.1)
        ax.plot(
            label_data[halfway:],
            reg.predict(label_data[halfway:].reshape(-1, 1)),
            color="red",
            label=f"$pred = {reg.coef_[0]:.2f} \\cdot x + {reg.intercept_:.2f}$",
        )
        
        ax.set_title(f"Predictions for {label_name}\nMSE: {mse:.4f}, R2: {r2:.4f}")
        ax.set_xlabel(f"Ground truth {label_name}")
        ax.set_ylabel(f"Predicted {label_name}")
        ax.legend()
    
    fig = plot_labels(plot_probe_predictions, f"Normalized linear probe predictions of {data_name} embeddings", 
                      list(labels.keys()), embeddings=embeddings, labels_dict=labels)
    
    fig.savefig(f"{plot_out_path}/{data_name}_pred.png", dpi=300)
    plt.show()
    plt.close(fig)

    # =============================================================================
    # Plot labels distribution
    # =============================================================================
    if "label" in labels or "log_label" in labels:
        key = "label" if "label" in labels else "log_label"
        test_embedings = embeddings[halfway:]
        test_labels = labels[key][halfway:]
        train_embeddings = embeddings[:halfway]
        train_labels = labels[key][:halfway]
        probe = LinearRegression().fit(train_embeddings, train_labels)
        predictions = probe.predict(test_embedings)
        
        fig, ax = plt.subplots(figsize=(6,4))
        unique_cross_sections = np.unique(test_labels)
        for cross_section in unique_cross_sections:
            mask = test_labels == cross_section
            predicted_label = predictions[mask]

            mean = np.mean(predicted_label)
            std = np.std(predicted_label)

            # plot histogram for the given class
            label = f"\\sigma_{{{cross_section:.2f}}}" if key == "label" else \
                    f"\\ln(\\sigma_{{{np.exp(cross_section):.2f}}}) = {cross_section:.2f}"
            n, bins, patches = ax.hist(predicted_label, bins=45, alpha=1/len(unique_cross_sections), 
                    label=f"${label}: {mean:.2f} \\pm {std:.2f}$")
            
            # add vertical line at the mean
            facecolor = patches[0].get_facecolor()
            ax.axvline(mean, linestyle="--", color=facecolor)

            # add floating text at the top of the vertical line with the same color
            ylim = ax.get_ylim()
            offset = 0.02 * (ylim[1] - ylim[0])
            y_top = min(n.max() + offset, ylim[1] - offset)
            ax.text(mean, y_top, f"{mean:.2f}", color=facecolor, ha="left", va="bottom", fontsize=9)

        ax.set_xlabel("Cross-section prediction")
        ax.set_ylabel("Count")
        ax.set_title(f"Cross-section distribution\nfor different labels in {data_name} model")
        ax.legend()

        fig.savefig(f"{plot_out_path}/{data_name}_cross_section_distribution.png", dpi=300)

        plt.show()
        plt.close(fig)
