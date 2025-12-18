"""
Fine-tuned vs Baseline Model Comparison and Visualization Module

This module compares the performance of fine-tuned AstroPT models against baseline
pre-trained models using linear probes. Generates visualizations of prediction distributions
and quantile comparisons across different condition classes (e.g., cross-sections).

Example:
    To compare fine-tuned and baseline AstroPT models on label predictions:
    
    $ python3 -m scripts.plots.finetune_comparison \\
        --finetuned_model_path model_weights/finetuned_model.pt \\
        --baseline_model_path model_weights/baseline_model.pt \\
        --labels label mass \\
        --nb_points 14000

"""

import numpy as np
import torch, argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from scripts.model_utils import compute_embeddings, load_astropt_model
from scripts.plots.plot_utils import set_fonts
from scripts.embeddings_utils import merge_datasets

from sklearn.linear_model import LinearRegression

def get_cross_section_stats(embeddings, labels, key):
    """
    Train linear probe and compute statistics for each cross-section class.
    
    Trains a linear regression probe on the first half of data and evaluates on the
    second half, computing median and percentile estimates (16% and 84%) for each
    unique value in the target label.
    
    Args:
        embeddings (np.ndarray): Embedding features of shape (num_samples, embedding_dim)
        labels (dict): Dictionary mapping label names to label arrays
        key (str): Label key to evaluate on (e.g., 'label', 'log_label')
    
    Returns:
        tuple:
            - stats (dict): Dictionary mapping unique label values to dicts containing:
                - 'median': Median predicted value
                - 'q16': 16% percentile
                - 'q84': 84% percentile
                - 'predictions': All predictions for this class
            - predictions (np.ndarray): Full array of test predictions
            - test_labels (np.ndarray): True test labels
    """
    halfway = len(embeddings) // 2
    
    test_embeddings = embeddings[halfway:]
    test_labels = labels[key][halfway:]
    train_embeddings = embeddings[:halfway]
    train_labels = labels[key][:halfway]
    
    probe = LinearRegression().fit(train_embeddings, train_labels)
    predictions = probe.predict(test_embeddings)
    
    unique_cross_sections = np.unique(test_labels)
    stats = {}
    
    for cross_section in unique_cross_sections:
        mask = test_labels == cross_section
        predicted_label = predictions[mask]
        
        median = np.median(predicted_label)
        q16 = np.percentile(predicted_label, 16)
        q84 = np.percentile(predicted_label, 84)
        
        stats[cross_section] = {
            'median': median,
            'q16': q16,
            'q84': q84,
            'predictions': predicted_label
        }
    
    return stats, predictions, test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='CompareAstroPT',
                    description='Compares finetuned vs baseline AstroPT on linear probe')
    
    parser.add_argument('--nb_points', type=int, default=14000)
    parser.add_argument('--labels', nargs='+', default=["label", "mass"])
    parser.add_argument('--finetuned_model_path', type=str, default="model_weights/finetuned_astropt.pt")
    parser.add_argument('--baseline_model_path', type=str, default="model_weights/baseline_astropt.pt")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    args = parser.parse_args()
    
    # Set plotting font size
    set_fonts()

    has_metals = torch.backends.mps.is_available()  
    device = torch.device('mps' if has_metals else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    print(f"Device: {device}")

    # Load finetuned model (your local weights)
    finetuned_model = load_astropt_model(args.finetuned_model_path, device=device, strict=False)
    baseline_model = load_astropt_model(args.baseline_model_path, device=device, strict=False)

    # Create dataset
    dataset = merge_datasets([
        "data/BAHAMAS/bahamas_0.1.pkl", 
        "data/BAHAMAS/bahamas_0.3.pkl", 
        "data/BAHAMAS/bahamas_1.pkl",
        "data/BAHAMAS/bahamas_cdm.pkl"], 
        args.labels, stack_features=False) \
        .shuffle(seed=42) \
        .take(args.nb_points)
    
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0 if has_metals else 4,
        prefetch_factor=None if has_metals else 3
    )

    # Compute embeddings for finetuned
    finetuned_embeddings, labels = compute_embeddings(finetuned_model, dl, device, args.labels)
    finetuned_embeddings = finetuned_embeddings.cpu().numpy()
    labels = {k: v.cpu().numpy() for k, v in labels.items()}

    # Compute embeddings for baseline
    baseline_embeddings, _ = compute_embeddings(baseline_model, dl, device, args.labels)
    baseline_embeddings = baseline_embeddings.cpu().numpy()

    # =============================================================================
    # Compare cross-section distributions
    # =============================================================================
    if "label" in labels or "log_label" in labels:
        key = "label" if "label" in labels else "log_label"
        
        finetuned_stats, _, test_labels = get_cross_section_stats(finetuned_embeddings, labels, key)
        baseline_stats, _, _ = get_cross_section_stats(baseline_embeddings, labels, key)
        
        unique_cross_sections = np.unique(test_labels)
        
        # Side-by-side histograms
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, stats, title in [
            (axes[0], baseline_stats, "Baseline AstroPT"),
            (axes[1], finetuned_stats, "Finetuned AstroPT")
        ]:
            for cross_section in unique_cross_sections:
                s = stats[cross_section]
                median, q16, q84 = s['median'], s['q16'], s['q84']
                
                label_str = f"\\sigma_{{{cross_section:.2f}}}"
                n, bins, patches = ax.hist(
                    s['predictions'], bins=45, alpha=1/len(unique_cross_sections),
                    label=f"${label_str}: {median:.2f}^{{+{q84-median:.2f}}}_{{-{median-q16:.2f}}}$"
                )
                ax.axvline(median, linestyle="--", color=patches[0].get_facecolor())
            
            ax.set_xlabel(r"Predicted $\sigma_{DM}/m$ [cm$^2$g$^{-1}$]")
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.legend()
        
        plt.suptitle(r"$\sigma_{DM}/m$ Linear Probe: Baseline vs Finetuned")
        plt.tight_layout()
        fig.savefig("figures/baseline_vs_finetuned_cross_section.png", dpi=300)
        plt.show()
        plt.close(fig)

        # Summary with error bars
        fig, ax = plt.subplots(figsize=(8, 6))
        
        true_values = np.array(sorted(unique_cross_sections))
        
        bl_medians = np.array([baseline_stats[cs]['median'] for cs in true_values])
        bl_lower = np.array([baseline_stats[cs]['median'] - baseline_stats[cs]['q16'] for cs in true_values])
        bl_upper = np.array([baseline_stats[cs]['q84'] - baseline_stats[cs]['median'] for cs in true_values])
        
        ft_medians = np.array([finetuned_stats[cs]['median'] for cs in true_values])
        ft_lower = np.array([finetuned_stats[cs]['median'] - finetuned_stats[cs]['q16'] for cs in true_values])
        ft_upper = np.array([finetuned_stats[cs]['q84'] - finetuned_stats[cs]['median'] for cs in true_values])
        
        x = np.arange(len(true_values))
        width = 0.35
        
        ax.errorbar(x - width/2, bl_medians, yerr=[bl_lower, bl_upper], 
                    fmt='o', capsize=5, label='Baseline AstroPT', color='tab:blue')
        ax.errorbar(x + width/2, ft_medians, yerr=[ft_lower, ft_upper], 
                    fmt='s', capsize=5, label='Finetuned AstroPT', color='tab:orange')
        
        ax.plot(x, true_values, 'k--', alpha=0.5, label='Ground truth')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{cs:.2f}" for cs in true_values])
        ax.set_xlabel(r"True $\sigma_{DM}/m$ [cm$^2$g$^{-1}$]")
        ax.set_ylabel(r"Predicted $\sigma_{DM}/m$ [cm$^2$g$^{-1}$]")
        ax.set_title(r"Linear Probe: Median $\pm$ 16/84% Quantiles")
        ax.legend()
        
        plt.tight_layout()
        fig.savefig("figures/baseline_vs_finetuned_summary.png", dpi=300)
        plt.show()
        plt.close(fig)