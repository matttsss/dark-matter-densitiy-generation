"""
Example script that samples from a trained astropt model and finetunes
embeddings on a linear probe
"""

import numpy as np
import torch, argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from scripts.model_utils import load_astropt_model
from scripts.plot_utils import plot_labels
from scripts.embedings_utils import merge_datasets, compute_embeddings

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
    args = parser.parse_args()
    
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

    embeddings, labels = compute_embeddings(model, dl, device, labels_name)
    embeddings = embeddings.cpu().numpy()
    labels = {k: v.cpu().numpy() for k, v in labels.items()}

    data_name = weights_filename.replace(".pt", "").split("/")[-1]
    print(f"Plotting probes for {data_name} model...")
    

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

    fig.savefig(f"figures/umap_{data_name}_magnitudes.png", dpi=300)
    plt.show()
    plt.clf()


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
    
    fig.savefig(f"figures/{data_name}_pred.png", dpi=300)
    plt.show()
    plt.clf()

    # =============================================================================
    # Plot labels distribution
    # =============================================================================
    test_embedings = embeddings[halfway:]
    test_labels = labels["label"][halfway:]
    train_embeddings = embeddings[:halfway]
    train_labels = labels["label"][:halfway]

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
        n, bins, patches = ax.hist(predicted_label, bins=45, alpha=1/len(unique_cross_sections), 
                label=f"$\\sigma_{{{cross_section:.2f}}}$: {mean:.2f} ± {std:.2f}")
        
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

    fig.savefig(f"figures/{data_name}_cross_section_distribution.png", dpi=300)

    plt.show()
    plt.clf()
