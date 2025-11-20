"""
Example script that samples from a trained astropt model and finetunes
embeddings on a linear probe
"""

import argparse

from plot_utils import plot_probes
from embedings_utils import merge_datasets, get_embeddings

from astropt.model_utils import load_astropt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PlotstroPT',
                    description='Generates plots for linear probe on astroPT embeddings')
    
    parser.add_argument('--astro_pt_data', action='store_true', default=False, help='Whether to use astroPT data for embeddings (default: False, dark data)')
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

    zss, labels_dict = get_embeddings(model, dataset, labels_name)

    data_name = "astroPT" if args.astro_pt_data else "dark"
    print(f"Plotting probes for {data_name} data...")
    plot_probes(zss, labels_dict, data_name)
