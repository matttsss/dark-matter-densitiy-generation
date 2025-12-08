import torch, argparse
import matplotlib.pyplot as plt

import numpy as np
from umap import UMAP
from torch.utils.data import DataLoader

from scripts.model_utils import LinearRegression, load_astropt_model, load_fm_model
from scripts.embedings_utils import merge_datasets, compute_embeddings

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

sampled_merged_label = {
    "mass": torch.asarray(14.6),
    "label": torch.asarray(0.1)
}
conditions = torch.stack([sampled_merged_label[label_name] for label_name in fm_model.config.conditions], dim=0).to(device)
conditions = conditions.unsqueeze(0).repeat(embeddings.size(0), 1)

fm_embeddings = fm_model.sample_flow(conditions, steps=500)

# ============== Predict and plot =================
lin_reg = LinearRegression(device).fit(embeddings, 
            torch.stack([labels[label_name] for label_name in fm_model.config.conditions], dim=1))
preds = lin_reg.predict(fm_embeddings).cpu().numpy()

plt.scatter(labels[fm_model.config.conditions[0]].cpu(), 
            labels[fm_model.config.conditions[1]].cpu(), alpha=0.1, label="True Embedding values")
plt.scatter(preds[:, 0], preds[:, 1], alpha=0.1, label="Generated Embedding predictions")
plt.scatter(sampled_merged_label[fm_model.config.conditions[0]].cpu(), 
            sampled_merged_label[fm_model.config.conditions[1]].cpu(), color='red', label='True Value', marker='x', s=100)
plt.xlabel(f"Predicted {fm_model.config.conditions[0]}")
plt.ylabel(f"Predicted {fm_model.config.conditions[1]}")
plt.title("Flow Matching Model Predictions")
plt.legend()
plt.grid()
plt.savefig("fm_model_validation.png")
plt.show()