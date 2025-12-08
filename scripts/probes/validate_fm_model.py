import torch, argparse
import matplotlib.pyplot as plt

import numpy as np
from umap import UMAP
from torch.utils.data import DataLoader

from scripts.model_utils import LinearRegression, load_astropt_model, load_fm_model
from scripts.embedings_utils import merge_datasets, compute_embeddings

def query_conditions(condition_names):
    condition_list = []

    print("Please enter the following condition values (Ctrl + C to cancel):")

    for name in condition_names:
        value = float(input(f"  {name}: "))
        condition_list.append(value)

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

    dl = DataLoader(
        dataset,
        batch_size = 64 if has_mps else 128,
        num_workers = 0 if has_mps else 4,
        prefetch_factor = None if has_mps else 3
    )

    # ============== Compute embeddings =================
    embeddings, labels = compute_embeddings(astropt_model, dl, device, merged_labels)
    lin_reg = LinearRegression(device).fit(embeddings, 
                torch.stack([labels[label_name] for label_name in fm_model.config.conditions], dim=1))


    # ============ UMAP embeddings plot =================
    umap_model = UMAP(n_components=2, random_state=42)
    embeddings_2d = umap_model.fit_transform(embeddings.cpu().numpy())

    while True:
        conditions = query_conditions(fm_model.config.conditions).to(device)
        conditions = conditions.unsqueeze(0).repeat(embeddings.size(0), 1)

        # ============== Predict and plot =================
        fm_embeddings = fm_model.sample_flow(conditions, steps=args.nb_steps)
        preds = lin_reg.predict(fm_embeddings).cpu().numpy()
        conditions = conditions.cpu().numpy()

        file_suffix = "_".join([f"{name}_{conditions[0,i]:.2f}" for i, name in enumerate(fm_model.config.conditions)]).replace(".", "_")


        mse = np.mean((preds - conditions)**2)

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get axis limits based on labels min/max
        label_vals_0 = labels[fm_model.config.conditions[0]].cpu().numpy()
        label_vals_1 = labels[fm_model.config.conditions[1]].cpu().numpy()
        x_min, x_max = label_vals_0.min(), label_vals_0.max()
        y_min, y_max = label_vals_1.min(), label_vals_1.max()
        
        # Create histogram with fixed range
        ax_colors = ax.hist2d(preds[:, 0], preds[:, 1], bins=60, cmap="viridis",
                              range=[[x_min, x_max], [y_min, y_max]])
        ax.scatter(conditions[0, 0], conditions[0, 1], 
                    color='red', label='True Value', marker='x', s=100, zorder=5)
        
        fig.colorbar(ax_colors[3], ax=ax, label="Density")
        ax.set_xlabel(f"{fm_model.config.conditions[0]}")
        ax.set_ylabel(f"{fm_model.config.conditions[1]}")
        ax.set_title("Flow Matching Model Predictions vs True Values\nMSE: {:.4f}".format(mse))
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"figures/flow_matching/inference/fm_gen_{file_suffix}.png")
        plt.show()
        plt.close(fig)


        # ============ UMAP embeddings plot =================
        fm_umap_embeddings = umap_model.transform(fm_embeddings.cpu().numpy())
        
        # Calculate shared axis limits
        x_min = min(embeddings_2d[:, 0].min(), fm_umap_embeddings[:, 0].min())
        x_max = max(embeddings_2d[:, 0].max(), fm_umap_embeddings[:, 0].max())
        y_min = min(embeddings_2d[:, 1].min(), fm_umap_embeddings[:, 1].min())
        y_max = max(embeddings_2d[:, 1].max(), fm_umap_embeddings[:, 1].max())
        
        fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
        axs[0].hist2d(embeddings_2d[:, 0], embeddings_2d[:, 1], bins=60, 
                      range=[[x_min, x_max], [y_min, y_max]], cmap="viridis")
        axs[0].set_title("UMAP Projection of AstroPT Embeddings")
        axs[0].set_xlabel("UMAP 1")
        axs[0].set_ylabel("UMAP 2")
        
        axs[1].hist2d(fm_umap_embeddings[:, 0], fm_umap_embeddings[:, 1], bins=60, 
                      range=[[x_min, x_max], [y_min, y_max]], cmap="viridis")
        axs[1].set_title("UMAP Projection of Flow Matching Embeddings")
        axs[1].set_xlabel("UMAP 1")

        fig.tight_layout()
        fig.savefig(f"figures/flow_matching/inference/fm_umap_{file_suffix}.png")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Validate Flow Matching Model")
    argparser.add_argument("--fm_model_path", type=str, required=True,
                            help="Path to the Flow Matching model to validate")
    argparser.add_argument("--astropt_model_path", type=str, required=True,
                            help="Path to the AstroPT model to use as reference")
    argparser.add_argument("--nb_steps", type=int, default=500,
                            help="Number of sampling steps for the Flow Matching model")
    argparser.add_argument("--nb_points", type=int, default=1000,
                            help="Number of points to use for validation")
    argparser.add_argument("--labels", nargs='+', default=["mass", "label"], help="Physical quantities to plot")
    args = argparser.parse_args()

    has_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if has_mps else 
                            "cuda" if torch.cuda.is_available() else "cpu")

    main(args, device)