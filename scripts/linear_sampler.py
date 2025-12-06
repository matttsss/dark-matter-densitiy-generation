import torch

from umap import UMAP
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from scripts.model_utils import LinearRegression, load_astropt_model
from scripts.embedings_utils import merge_datasets, compute_embeddings

if __name__ == "__main__":
    import argparse
 
    argparser = argparse.ArgumentParser()
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
    astropt_model = load_astropt_model(args.astropt_model_path, device=device, strict=True)
    astropt_model.eval()

    # =============== Load datasets ===================
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
            .select_columns(["images", "images_positions", *args.labels]) \
            .shuffle(seed=42) \
            .take(args.nb_points)
    
    dl = DataLoader(
        dataset,
        batch_size = 64 if has_mps else 128,
        num_workers = 0 if has_mps else 4,
        prefetch_factor = None if has_mps else 3
    )

    # ============== Compute embeddings =================
    embeddings, labels = compute_embeddings(astropt_model, dl, device, args.labels)
    labels = torch.stack([labels[k] for k in args.labels], dim=-1)

    with torch.no_grad():
        lin_reg = LinearRegression(device=device).fit(embeddings, labels)

        embeddings = embeddings.cpu().numpy()
        sampled_embeddings = lin_reg.sample(labels).cpu().numpy()
        labels = labels.cpu().numpy()
    
    # ============== Plot comparisons =================
    umap = UMAP(n_components=2, random_state=42)
    embeddings_2d = umap.fit_transform(embeddings)
    sampled_embeddings_2d = umap.transform(sampled_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', s=5, alpha=0.5)
    axes[0].set_title("Original Embeddings UMAP")
    axes[1].scatter(sampled_embeddings_2d[:, 0], sampled_embeddings_2d[:, 1], c='red', s=5, alpha=0.5)
    axes[1].set_title("Sampled Embeddings UMAP")

    plt.show()

