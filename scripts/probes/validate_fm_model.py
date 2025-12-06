import torch
import matplotlib.pyplot as plt

import numpy as np

from umap import UMAP
from scripts.model_utils import LinearRegression
from scripts.plot_utils import plot_labels

@torch.no_grad()
def predict_fm_model(fm_model, embeddings, labels, cond_names, label_names, train_ratio=0.8):
    cond = torch.stack([labels[k] for k in fm_model.config.conditions], dim=1)
    labels_plot = torch.stack([labels[k] for k in label_names], dim=1)

    nb_train = int(train_ratio * embeddings.size(0))
    lin_reg = LinearRegression(embeddings.device)
    lin_reg.fit(embeddings[:nb_train], labels_plot[:nb_train])

    lin_preds = lin_reg.predict(embeddings)
    lin_preds = {label_name: lin_preds[:, i] for i, label_name in enumerate(label_names)}

    vf_embeddings = fm_model.sample_flow(cond, steps=500)
    vf_preds = lin_reg.predict(vf_embeddings)
    vf_preds = {cond_name: vf_preds[:, i] for i, cond_name in enumerate(label_names)}

    return vf_embeddings, vf_preds, lin_preds

def umap_compare(vf_embeddings, embeddings, ground_truth):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    umap_proj = UMAP(n_components=2, random_state=42)
    ref_embeddings_2d = umap_proj.fit_transform(embeddings)
    fm_embeddings_2d = umap_proj.transform(vf_embeddings)

    sc0 = axs[0].scatter(ref_embeddings_2d[:, 0], ref_embeddings_2d[:, 1], 
                        c=ground_truth["mass"], alpha=0.3)
    axs[0].set_title("Reference Embeddings UMAP Projection")
    fig.colorbar(sc0, ax=axs[0], label='Mass')

    distance = np.linalg.norm(embeddings - vf_embeddings, axis=1)
    sc1 = axs[1].scatter(fm_embeddings_2d[:, 0], fm_embeddings_2d[:, 1], 
                         c=distance, alpha=0.3)
    axs[1].set_title("Flow Matching Embeddings UMAP Projection")
    fig.colorbar(sc1, ax=axs[1], label='Distance to Reference Embedding')
    
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    import argparse
    from scripts.model_utils import load_astropt_model, load_fm_model
    from scripts.embedings_utils import merge_datasets, compute_embeddings
    from torch.utils.data import DataLoader
 
    argparser = argparse.ArgumentParser(description="Validate Flow Matching Model")
    argparser.add_argument("--fm_model_path", type=str, required=True,
                           help="Path to the Flow Matching model to validate")
    argparser.add_argument("--astropt_model_path", type=str, required=True,
                           help="Path to the AstroPT model to use as reference")
    argparser.add_argument("--nb_points", type=int, default=1000,
                           help="Number of points to use for validation")
    argparser.add_argument("--labels", nargs='+', default=["mass", "label"], help="Physical quantities to plot")
    args = argparser.parse_args()

    has_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if has_mps else 
                          "cuda" if torch.cuda.is_available() else "cpu")

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
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
            .select_columns(["images", "images_positions", *merged_labels]) \
            .shuffle(seed=42) \
            .take(args.nb_points)
    
    dl = DataLoader(
        dataset,
        batch_size = 64 if has_mps else 128,
        num_workers = 0 if has_mps else 4,
        prefetch_factor = None if has_mps else 3
    )

    # ============== Compute embeddings =================
    embeddings, labels = compute_embeddings(astropt_model, dl, device, merged_labels)

    vf_embeddings, vf_preds, lin_preds = predict_fm_model(
        fm_model, embeddings, labels, fm_model.config.conditions, args.labels)
    
    vf_embeddings = vf_embeddings.cpu().numpy()
    embeddings = embeddings.cpu().numpy()
    ground_truth = {k: labels[k].cpu().numpy() for k in args.labels}
    lin_preds = {k: lin_preds[k].cpu().numpy() for k in args.labels}
    vf_preds = {k: vf_preds[k].cpu().numpy() for k in args.labels}


    def plot_func(fig, ax, label_name):
        rel_diff = np.abs(lin_preds[label_name] - ground_truth[label_name]) / np.abs(ground_truth[label_name])

        ax.scatter(ground_truth[label_name], lin_preds[label_name], alpha=0.1, label='Linear Regression Predictions')
        z = ax.scatter(ground_truth[label_name], vf_preds[label_name], 
                       c=rel_diff, alpha=0.1, label='Flow Matching Predictions')
        
        fig.colorbar(z, ax=ax, label='Prediction Difference Norm')
        ax.set_title(f"Predictions for {label_name}")

    fig = plot_labels(plot_func, "Flow Matching vs Linear Regression Predictions", args.labels)
    fig.savefig("figures/fm_predictions.png", dpi=300)
    fig.show()
    plt.close()

    fig = umap_compare(vf_embeddings, embeddings, ground_truth)
    fig.savefig("figures/umap_comparison.png")
    fig.show()
    plt.close()
