"""
Example script that samples from a trained astropt model and finetunes
embeddings on a linear probe
"""

import torch, argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from embedings_utils import merge_datasets, batch_to_device
from plot_utils import plot_labels
from model_utils import load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PlotstroPT',
                    description='Generates plots for linear probe on astroPT embeddings')
    
    parser.add_argument('--nb_points', type=int, default=1000, help='Number of points to use for embeddings')
    parser.add_argument('--labels', nargs='+', default=["mass"], help='Labels to use for embeddings')
    args = parser.parse_args()
    
    labels_name = args.labels    
    weights_filename = "lora_32_5_labels.pt"
    
    print("Loading/generating embeddings...")
 
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
            .select_columns(["images", "images_positions", *labels_name]) \
            .shuffle(seed=42) \
            .take(args.nb_points)
    
    has_metals = torch.backends.mps.is_available()  
    device = torch.device('mps' if has_metals else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    
    print(f"Generating embeddings on device: {device}")

    # =============================================================================
    # ================== Load model and generate embeddings =======================
    # =============================================================================

    model = load_model(checkpoint_path=f"model/{weights_filename}",
                       device=device,
                       lora_rank=32,
                       output_dim=len(labels_name))
    model.eval()

    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 200,
        num_workers = 0 if has_metals else 2,
        prefetch_factor = None if has_metals else 3
    )

    labels = []
    embeddings = []
    predictions = []
    with torch.no_grad():
        for B in tqdm(dl, desc="Generating embeddings"):
            B = batch_to_device(B, device)

            batch_labels = torch.stack([B[label] for label in labels_name], dim=1)
            head_value = model.generate_embeddings(B, reduction='none')["images"]
            batch_predictions = model.task_head(head_value)

            embeddings.append(torch.mean(head_value, dim=1).cpu())
            labels.append(batch_labels.cpu())
            predictions.append(batch_predictions.cpu())
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    predictions = torch.cat(predictions, dim=0).numpy()
    labels = {label_name: labels[:, i] for i, label_name in enumerate(labels_name)}
    predictions = {label_name: predictions[:, i] for i, label_name in enumerate(labels_name)}


    # =============================================================================
    # ======================== Plot UMAP projections ==============================
    # =============================================================================

    data_name = weights_filename.replace(".pt", "")
    print(f"Plotting probes for {data_name} data...")

    umap = UMAP(n_components=2)
    umap_embeddings = StandardScaler().fit_transform(embeddings)
    umap_embeddings = umap.fit_transform(umap_embeddings)

    def plot_probe(fig, ax, label_name, umap_embeddings, labels_dict):
        data = labels_dict[label_name]
        vmax = np.percentile(data, 95)
        sc = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=data, vmax=vmax, cmap="viridis")

        ax.set_title(label_name)
        fig.colorbar(sc, ax=ax, label=label_name)

    fig = plot_labels(plot_probe, f"UMAP projection of {data_name} embeddings", labels_name,
                      umap_embeddings=umap_embeddings, labels_dict=labels)
    
    fig.savefig(f"figures/umap_{data_name}_magnitudes.png", dpi=300)
    plt.show()
    plt.clf()


    # =============================================================================
    # Plot Probe predictions
    # =============================================================================

    def plot_probe_predictions(fig, ax, label_name, embeddings, labels_dict):
        label_data = labels_dict[label_name]

        # Normalize label data
        # mean, std = label_data.mean(), label_data.std()
        # label_data = (label_data - mean) / std

        # Train linear probe to predict label from embeddings
        halfway = len(embeddings) // 2
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
                      labels_name, embeddings=embeddings, labels_dict=labels)
    
    fig.savefig(f"figures/{data_name}_pred.png", dpi=300)
    plt.show()
    plt.clf()

    # =============================================================================
    # Plot labels distribution
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6,4))
    unique_cross_sections = np.unique(labels["label"])
    for cross_section in unique_cross_sections:
        mask = labels["label"] == cross_section
        ax.hist(predictions["label"][mask], bins=45, alpha=0.5, label=f"Label {cross_section:.3f}")
    ax.set_xlabel("Mass")
    ax.set_ylabel("Count")
    ax.set_title(f"Mass distribution for different labels in {data_name} data")
    ax.legend()
    fig.savefig(f"figures/{data_name}_mass_distribution.png", dpi=300)
    plt.show()
    plt.clf()