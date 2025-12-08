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

from scripts.embedings_utils import merge_datasets, batch_to_device
from scripts.plot_utils import plot_labels
from scripts.model_utils import load_astropt_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PlotstroPT',
                    description='Generates plots for linear probe on astroPT embeddings')
    
    parser.add_argument('--nb_points', type=int, default=1000, help='Number of points to use for embeddings')
    parser.add_argument('--model_path', type=str, default="model/ckpt.pt", help='Path to the model checkpoint')
    args = parser.parse_args()

    has_metals = torch.backends.mps.is_available()  
    device = torch.device('mps' if has_metals else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    
    print(f"Generating embeddings on device: {device}")
    weights_filename = args.model_path

    model, label_names = load_astropt_model(checkpoint_path=weights_filename, 
                       device=device, get_label_names=True, strict=True)
    model.eval()

     
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"], 
        label_names, stack_features=False)\
            .shuffle(seed=42) \
            .take(args.nb_points)

    # =============================================================================
    # ================== Load model and generate embeddings =======================
    # =============================================================================

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

            batch_labels = torch.stack([B[label] for label in label_names], dim=1)
            head_value = model.generate_embeddings(B, reduction='none')["images"]
            batch_predictions = model.task_head(head_value)

            embeddings.append(torch.mean(head_value, dim=1).cpu())
            labels.append(batch_labels.cpu())
            predictions.append(batch_predictions.cpu())
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    predictions = torch.cat(predictions, dim=0).numpy()
    labels = {label_name: labels[:, i] for i, label_name in enumerate(label_names)}
    predictions = {label_name: predictions[:, i] for i, label_name in enumerate(label_names)}


    # =============================================================================
    # ======================== Plot UMAP projections ==============================
    # =============================================================================

    data_name = weights_filename.replace(".pt", "").split("/")[-1]
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

    fig = plot_labels(plot_probe, f"UMAP projection of {data_name} embeddings", label_names,
                      umap_embeddings=umap_embeddings, labels_dict=labels)
    
    fig.savefig(f"figures/umap_{data_name}_magnitudes.png", dpi=300)
    plt.show()
    plt.clf()


    # =============================================================================
    # Plot Probe predictions
    # =============================================================================

    def plot_probe_predictions(fig, ax, label_name, ref_dict, preds_dict):
        ref_data = ref_dict[label_name]
        pred_data = preds_dict[label_name]

        # Normalize label data
        mean, std = ref_data.mean(), ref_data.std()
        ref_data = (ref_data - mean) / std
        pred_data = (pred_data - mean) / std

        # Compute metrics
        mse = mean_squared_error(ref_data, pred_data)
        r2 = r2_score(ref_data, pred_data)

        # Plot predictions vs ground truth
        reg = LinearRegression().fit(ref_data.reshape(-1, 1), pred_data)
        ax.scatter(ref_data, pred_data, alpha=0.1)
        ax.plot(
            ref_data,
            reg.predict(ref_data.reshape(-1, 1)),
            color="red",
            label=f"$pred = {reg.coef_[0]:.2f} \\cdot x + {reg.intercept_:.2f}$",
        )
        
        ax.set_title(f"Predictions for {label_name}\nMSE: {mse:.4f}, R2: {r2:.4f}")
        ax.set_xlabel(f"Ground truth {label_name}")
        ax.set_ylabel(f"Predicted {label_name}")
        ax.legend()

    
    fig = plot_labels(plot_probe_predictions, f"Normalized linear probe predictions of {data_name} embeddings", 
                      label_names, preds_dict=predictions, ref_dict=labels)
    
    fig.savefig(f"figures/{data_name}_pred.png", dpi=300)
    plt.show()
    plt.clf()

    # =============================================================================
    # Plot labels distribution
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6,4))
    label_ref = labels["label"]
    label_preds = predictions["label"]
    unique_cross_sections = np.unique(label_ref)
    for cross_section in unique_cross_sections:
        mask = label_ref == cross_section
        predicted_label = label_preds[mask]

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