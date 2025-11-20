"""
Example script that samples from a trained astropt model and finetunes
embeddings on a linear probe
"""

import torch, argparse
from astropt.model_utils import load_astropt
from torch.utils.data import DataLoader

from plot_utils import plot_probes
from embedings_utils import merge_datasets, compute_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PlotstroPT',
                    description='Generates plots for linear probe on astroPT embeddings')
    
    parser.add_argument('--nb_points', type=int, default=1000, help='Number of points to use for embeddings')
    parser.add_argument('--labels', nargs='+', default=["mass"], help='Labels to use for embeddings')
    args = parser.parse_args()
    
    labels_name = args.labels    
    print("Loading/generating embeddings...")

    model = load_astropt("Smith42/astroPT_v2.0", 
                         path="astropt/095M")

    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl"])

    dataset = dataset.select_columns(["images", "images_positions", *labels_name]) \
            .shuffle(seed=42) \
            .take(args.nb_points)
    
    has_metals = torch.backends.mps.is_available()  
    if has_metals:
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Generating embeddings on device: {device}")

    model = model.to(device)

    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 128,
        num_workers = 0 if has_metals else 10,
        prefetch_factor = None if has_metals else 3
    )

    embeddings, labels = compute_embeddings(model, dl, device, labels_name)
    embeddings = embeddings.cpu().numpy()
    labels = {k: v.cpu().numpy() for k, v in labels.items()}

    data_name = "astroPT" if args.astro_pt_data else "dark"
    print(f"Plotting probes for {data_name} data...")
    plot_probes(embeddings, labels, data_name)
