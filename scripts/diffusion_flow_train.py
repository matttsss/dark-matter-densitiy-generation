# scripts/diffusion_train_fm.py
#
# Training script for Flow Matching → DDPM pipeline
# Generates images conditioned on (mass, label) via FM embeddings
#
# Key insight: This is a GENERATIVE model, not reconstruction.
# We evaluate by comparing DISTRIBUTIONS, not paired samples.

import copy
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functools import partial
import pickle

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, skew, kurtosis, ks_2samp
from scipy import ndimage
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

from scripts.model_utils import load_fm_model
from generative_model.DDPM import DDPM

LABEL_NAMES = ["mass", "label"]
DATASETS = [
    "data/BAHAMAS/bahamas_0.1.pkl",
    "data/BAHAMAS/bahamas_0.3.pkl",
    "data/BAHAMAS/bahamas_1.pkl",
    "data/BAHAMAS/bahamas_cdm.pkl",
]


# =============================================================================
# POWER SPECTRUM UTILITIES
# =============================================================================

def compute_power_spectrum_2d(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute radially-averaged 2D power spectrum."""
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift)**2
    
    H, W = image.shape
    center = H // 2
    
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    n_bins = H // 2
    ell_edges = np.linspace(0, n_bins, n_bins + 1)
    ell_centers = (ell_edges[:-1] + ell_edges[1:]) / 2
    
    P_ell = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= ell_edges[i]) & (r < ell_edges[i + 1])
        if mask.sum() > 0:
            P_ell[i] = power[mask].mean()
    
    return ell_centers, P_ell


# =============================================================================
# FM-APPROPRIATE METRICS (DISTRIBUTIONAL, NOT PAIRED)
# =============================================================================

def compute_psd_distribution_metrics(real_images: torch.Tensor, 
                                      generated_images: torch.Tensor) -> dict:
    """
    Compare power spectrum DISTRIBUTIONS between real and generated.
    This is the correct metric for generative models - NOT cross-correlation.
    """
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    
    if not np.isfinite(gen_np).all():
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    if real_np.ndim == 4:
        real_np = real_np[:, 0]
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    # Compute PSDs for all images
    real_psds, gen_psds = [], []
    ell = None
    
    for i in range(real_np.shape[0]):
        ell_i, psd = compute_power_spectrum_2d(real_np[i])
        if ell is None:
            ell = ell_i
        real_psds.append(psd)
    
    for i in range(gen_np.shape[0]):
        _, psd = compute_power_spectrum_2d(gen_np[i])
        gen_psds.append(psd)
    
    real_psds = np.array(real_psds)
    gen_psds = np.array(gen_psds)
    
    # Mean and std PSDs
    mean_real_psd = real_psds.mean(axis=0)
    mean_gen_psd = gen_psds.mean(axis=0)
    std_real_psd = real_psds.std(axis=0)
    std_gen_psd = gen_psds.std(axis=0)
    
    # PSD ratio (should be ~1 if distributions match)
    psd_ratio = mean_gen_psd / (mean_real_psd + 1e-10)
    log_psd_ratio = np.log10(psd_ratio + 1e-10)
    
    n = len(ell)
    low, mid, high = slice(1, n//3), slice(n//3, 2*n//3), slice(2*n//3, n)
    
    return {
        # PSD ratio (1.0 = perfect match)
        "psd_ratio_low": float(psd_ratio[low].mean()),
        "psd_ratio_mid": float(psd_ratio[mid].mean()),
        "psd_ratio_high": float(psd_ratio[high].mean()),
        "psd_ratio_mean": float(psd_ratio[1:].mean()),
        # Log PSD ratio (0.0 = perfect)
        "log_psd_mae": float(np.abs(log_psd_ratio[1:]).mean()),
        # Variance comparison
        "psd_std_ratio": float((std_gen_psd / (std_real_psd + 1e-10))[1:].mean()),
        # For plotting
        "_ell": ell,
        "_mean_real_psd": mean_real_psd,
        "_mean_gen_psd": mean_gen_psd,
        "_std_real_psd": std_real_psd,
        "_std_gen_psd": std_gen_psd,
        "_psd_ratio": psd_ratio,
    }


def compute_summary_statistics(real_images: torch.Tensor,
                                generated_images: torch.Tensor) -> dict:
    """Compare per-image summary statistics distributions."""
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    
    if not np.isfinite(gen_np).all():
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    if real_np.ndim == 4:
        real_np = real_np[:, 0]
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    def extract_stats(images):
        stats = {'flux': [], 'max': [], 'std': [], 'skew': [], 'kurt': [], 'n_peaks': []}
        for img in images:
            stats['flux'].append(img.sum())
            stats['max'].append(img.max())
            stats['std'].append(img.std())
            stats['skew'].append(skew(img.flatten()))
            stats['kurt'].append(kurtosis(img.flatten()))
            threshold = img.mean() + 3 * img.std()
            _, n_peaks = ndimage.label(img > threshold)
            stats['n_peaks'].append(n_peaks)
        return {k: np.array(v) for k, v in stats.items()}
    
    real_stats = extract_stats(real_np)
    gen_stats = extract_stats(gen_np)
    
    metrics = {}
    for name in real_stats.keys():
        real_vals = real_stats[name][np.isfinite(real_stats[name])]
        gen_vals = gen_stats[name][np.isfinite(gen_stats[name])]
        
        if len(real_vals) > 0 and len(gen_vals) > 0:
            metrics[f"{name}_mean_real"] = float(real_vals.mean())
            metrics[f"{name}_mean_gen"] = float(gen_vals.mean())
            metrics[f"{name}_ratio"] = float(gen_vals.mean() / (real_vals.mean() + 1e-10))
            ks_stat, _ = ks_2samp(real_vals, gen_vals)
            metrics[f"{name}_ks"] = float(ks_stat)
    
    return metrics


def compute_conditioning_fidelity(generated_images: torch.Tensor,
                                   masses: torch.Tensor,
                                   labels: torch.Tensor,
                                   real_images: torch.Tensor = None) -> dict:
    """
    THE KEY METRIC: Can we recover input conditions from generated images?
    
    This tests the entire FM → DDPM pipeline:
    - FM encodes (mass, label) into embedding
    - DDPM generates image from embedding
    - If we can recover mass/label from image, conditioning worked!
    """
    gen_np = generated_images.cpu().numpy()
    masses_np = masses.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    if not np.isfinite(gen_np).all():
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    def extract_features(images):
        features = []
        for img in images:
            flux = img.sum()
            max_val = img.max()
            std_val = img.std()
            
            # Peak statistics
            _, n_peaks_95 = ndimage.label(img > np.percentile(img, 95))
            _, n_peaks_99 = ndimage.label(img > np.percentile(img, 99))
            
            # Concentration
            h, w = img.shape
            center_flux = img[h//4:3*h//4, w//4:3*w//4].sum()
            concentration = center_flux / (flux + 1e-10)
            
            # Morphology
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            total = flux + 1e-10
            cy = (y_coords * img).sum() / total
            cx = (x_coords * img).sum() / total
            var_y = ((y_coords - cy)**2 * img).sum() / total
            var_x = ((x_coords - cx)**2 * img).sum() / total
            ellipticity = abs(var_y - var_x) / (var_y + var_x + 1e-10)
            
            feat = [flux, max_val, std_val, n_peaks_95, n_peaks_99, 
                    concentration, ellipticity, skew(img.flatten()), kurtosis(img.flatten())]
            feat = [0.0 if not np.isfinite(f) else f for f in feat]
            features.append(feat)
        return np.array(features)
    
    gen_features = extract_features(gen_np)
    metrics = {}
    
    # === MASS RECOVERY ===
    # Direct flux-mass correlation (THE most important metric!)
    gen_flux = gen_np.sum(axis=(1, 2))
    flux_mass_corr, _ = pearsonr(gen_flux, masses_np)
    metrics["flux_mass_corr"] = float(flux_mass_corr) if np.isfinite(flux_mass_corr) else 0.0
    
    # Regression-based mass recovery
    mass_reg = Ridge(alpha=1.0)
    mass_reg.fit(gen_features, masses_np)
    pred_mass = mass_reg.predict(gen_features)
    metrics["mass_r2"] = float(r2_score(masses_np, pred_mass))
    
    # === LABEL RECOVERY ===
    unique_labels = np.unique(labels_np)
    if len(unique_labels) > 1:
        label_reg = Ridge(alpha=1.0)
        label_reg.fit(gen_features, labels_np)
        pred_label = label_reg.predict(gen_features)
        metrics["label_r2"] = float(r2_score(labels_np, pred_label))
        
        # Classification accuracy
        if len(unique_labels) <= 10:
            try:
                clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
                clf.fit(gen_features, labels_np.astype(int))
                pred_class = clf.predict(gen_features)
                metrics["label_acc"] = float(accuracy_score(labels_np.astype(int), pred_class))
            except:
                metrics["label_acc"] = 0.0
    
    # === BASELINE COMPARISON ===
    if real_images is not None:
        real_np = real_images.cpu().numpy()
        if real_np.ndim == 4:
            real_np = real_np[:, 0]
        
        real_features = extract_features(real_np)
        real_flux = real_np.sum(axis=(1, 2))
        
        # Real flux-mass correlation
        real_corr, _ = pearsonr(real_flux, masses_np)
        metrics["flux_mass_corr_real"] = float(real_corr) if np.isfinite(real_corr) else 0.0
        
        # Real mass R²
        mass_reg_real = Ridge(alpha=1.0)
        mass_reg_real.fit(real_features, masses_np)
        metrics["mass_r2_real"] = float(r2_score(masses_np, mass_reg_real.predict(real_features)))
        
        # Retention ratio (how much info is preserved)
        metrics["mass_r2_retention"] = float(metrics["mass_r2"] / (metrics["mass_r2_real"] + 1e-10))
    
    return metrics


def compute_all_metrics(real_images: torch.Tensor,
                         generated_images: torch.Tensor,
                         masses: torch.Tensor,
                         labels: torch.Tensor) -> dict:
    """Compute all FM-appropriate metrics."""
    metrics = {}
    
    # 1. PSD distribution comparison
    psd = compute_psd_distribution_metrics(real_images, generated_images)
    metrics.update({f"psd/{k}": v for k, v in psd.items() if not k.startswith("_")})
    metrics["_psd_raw"] = psd
    
    # 2. Summary statistics comparison
    summary = compute_summary_statistics(real_images, generated_images)
    metrics.update({f"stats/{k}": v for k, v in summary.items()})
    
    # 3. Conditioning fidelity (KEY!)
    cond = compute_conditioning_fidelity(generated_images, masses, labels, real_images)
    metrics.update({f"cond/{k}": v for k, v in cond.items()})
    
    return metrics


# =============================================================================
# VISUALIZATION (FM-APPROPRIATE)
# =============================================================================

def plot_psd_comparison(metrics: dict, title: str = "") -> plt.Figure:
    """Plot PSD comparison - correct for generative models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    raw = metrics.get("_psd_raw", metrics)
    ell = raw["_ell"]
    mean_real = raw["_mean_real_psd"]
    mean_gen = raw["_mean_gen_psd"]
    std_real = raw["_std_real_psd"]
    std_gen = raw["_std_gen_psd"]
    psd_ratio = raw["_psd_ratio"]
    
    # Plot 1: PSD comparison with bands
    ax = axes[0]
    ax.loglog(ell[1:], mean_real[1:], 'b-', lw=2, label='Real')
    ax.fill_between(ell[1:], mean_real[1:] - std_real[1:], mean_real[1:] + std_real[1:], 
                    alpha=0.3, color='blue')
    ax.loglog(ell[1:], mean_gen[1:], 'r--', lw=2, label='Generated')
    ax.fill_between(ell[1:], mean_gen[1:] - std_gen[1:], mean_gen[1:] + std_gen[1:], 
                    alpha=0.3, color='red')
    ax.set_xlabel("ℓ")
    ax.set_ylabel("P(ℓ)")
    ax.set_title(f"Power Spectrum {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PSD ratio
    ax = axes[1]
    ax.semilogx(ell[1:], psd_ratio[1:], 'k-', lw=2)
    ax.axhline(1.0, color='gray', ls='--', label='Perfect')
    ax.fill_between(ell[1:], 0.9, 1.1, alpha=0.2, color='green', label='±10%')
    ax.set_xlabel("ℓ")
    ax.set_ylabel("P_gen / P_real")
    ax.set_title("PSD Ratio")
    ax.set_ylim(0.5, 1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_generated_samples(generated_images: torch.Tensor, 
                            masses: torch.Tensor,
                            labels: torch.Tensor,
                            n_samples: int = 8) -> plt.Figure:
    """Show generated samples sorted by mass (no misleading 'real' comparison)."""
    gen_np = generated_images.cpu().numpy()
    masses_np = masses.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    sorted_idx = np.argsort(masses_np)
    sample_idx = np.linspace(0, len(sorted_idx) - 1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(2, n_samples // 2, figsize=(3 * n_samples // 2, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_idx):
        actual_idx = sorted_idx[idx]
        img = gen_np[actual_idx]
        mass = masses_np[actual_idx]
        label = labels_np[actual_idx]
        
        axes[i].imshow(img, cmap='viridis')
        axes[i].set_title(f"M={mass:.2f}, L={label:.0f}")
        axes[i].axis('off')
    
    plt.suptitle("Generated Samples (sorted by mass)")
    plt.tight_layout()
    return fig


def plot_conditioning_quality(generated_images: torch.Tensor,
                               masses: torch.Tensor,
                               labels: torch.Tensor,
                               real_images: torch.Tensor = None) -> plt.Figure:
    """The key diagnostic: Does conditioning work?"""
    gen_np = generated_images.cpu().numpy()
    masses_np = masses.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    gen_flux = gen_np.sum(axis=(1, 2))
    corr_gen, _ = pearsonr(masses_np, gen_flux)
    
    n_plots = 3 if real_images is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    # Plot 1: Flux vs Mass (generated)
    ax = axes[0]
    scatter = ax.scatter(masses_np, gen_flux, c=labels_np, alpha=0.6, s=30, cmap='viridis')
    ax.set_xlabel("Input Mass")
    ax.set_ylabel("Generated Flux")
    ax.set_title(f"Generated: r={corr_gen:.3f}")
    plt.colorbar(scatter, ax=ax, label='Label')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Flux by label
    ax = axes[1]
    for lbl in np.unique(labels_np):
        mask = labels_np == lbl
        ax.hist(gen_flux[mask], bins=20, alpha=0.5, label=f'Label {lbl:.0f}')
    ax.set_xlabel("Generated Flux")
    ax.set_ylabel("Count")
    ax.set_title("Flux Distribution by Label")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Compare with real
    if real_images is not None:
        ax = axes[2]
        real_np = real_images.cpu().numpy()
        if real_np.ndim == 4:
            real_np = real_np[:, 0]
        real_flux = real_np.sum(axis=(1, 2))
        corr_real, _ = pearsonr(masses_np, real_flux)
        
        ax.scatter(masses_np, real_flux, alpha=0.4, s=30, label=f'Real (r={corr_real:.3f})', c='blue')
        ax.scatter(masses_np, gen_flux, alpha=0.4, s=30, label=f'Gen (r={corr_gen:.3f})', c='red')
        ax.set_xlabel("Mass")
        ax.set_ylabel("Flux")
        ax.set_title("Flux-Mass Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_distribution_comparison(real_images: torch.Tensor,
                                  generated_images: torch.Tensor) -> plt.Figure:
    """Compare distributions between real and generated."""
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    
    if real_np.ndim == 4:
        real_np = real_np[:, 0]
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Pixel histogram
    ax = axes[0, 0]
    ax.hist(real_np.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax.hist(gen_np.flatten(), bins=50, alpha=0.5, label='Gen', density=True)
    ax.set_xlabel("Pixel Value")
    ax.set_title("Pixel Distribution")
    ax.legend()
    ax.set_yscale('log')
    
    # Flux histogram
    ax = axes[0, 1]
    real_flux = real_np.sum(axis=(1, 2))
    gen_flux = gen_np.sum(axis=(1, 2))
    ax.hist(real_flux, bins=30, alpha=0.5, label='Real', density=True)
    ax.hist(gen_flux, bins=30, alpha=0.5, label='Gen', density=True)
    ax.set_xlabel("Total Flux")
    ax.set_title("Flux Distribution")
    ax.legend()
    
    # Max histogram
    ax = axes[0, 2]
    ax.hist(real_np.max(axis=(1, 2)), bins=30, alpha=0.5, label='Real', density=True)
    ax.hist(gen_np.max(axis=(1, 2)), bins=30, alpha=0.5, label='Gen', density=True)
    ax.set_xlabel("Max Pixel")
    ax.set_title("Peak Brightness")
    ax.legend()
    
    # Skewness
    ax = axes[1, 0]
    real_skew = [skew(img.flatten()) for img in real_np]
    gen_skew = [skew(img.flatten()) for img in gen_np]
    ax.hist(real_skew, bins=30, alpha=0.5, label='Real', density=True)
    ax.hist(gen_skew, bins=30, alpha=0.5, label='Gen', density=True)
    ax.set_xlabel("Skewness")
    ax.set_title("Skewness Distribution")
    ax.legend()
    
    # Peak count
    ax = axes[1, 1]
    def count_peaks(img):
        _, n = ndimage.label(img > img.mean() + 3 * img.std())
        return n
    real_peaks = [count_peaks(img) for img in real_np]
    gen_peaks = [count_peaks(img) for img in gen_np]
    max_peak = max(max(real_peaks), max(gen_peaks)) + 2
    ax.hist(real_peaks, bins=range(0, max_peak), alpha=0.5, label='Real', density=True, align='left')
    ax.hist(gen_peaks, bins=range(0, max_peak), alpha=0.5, label='Gen', density=True, align='left')
    ax.set_xlabel("Number of Peaks")
    ax.set_title("Peak Count")
    ax.legend()
    
    # Q-Q plot
    ax = axes[1, 2]
    n_points = 100
    real_q = np.percentile(real_flux, np.linspace(0, 100, n_points))
    gen_q = np.percentile(gen_flux, np.linspace(0, 100, n_points))
    ax.scatter(real_q, gen_q, alpha=0.5, s=10)
    ax.plot([real_q.min(), real_q.max()], [real_q.min(), real_q.max()], 'r--')
    ax.set_xlabel("Real Flux Quantiles")
    ax.set_ylabel("Gen Flux Quantiles")
    ax.set_title("Q-Q Plot (Flux)")
    
    plt.tight_layout()
    return fig


# =============================================================================
# DATA LOADING
# =============================================================================

def load_images_with_features():
    """Load raw images and features from pickle files."""
    torchify = lambda x: torch.from_numpy(np.asarray(x)).float()
    
    all_images, all_mass, all_label = [], [], []
    
    for dataset_path in DATASETS:
        print(f"Loading {dataset_path}...")
        with open(dataset_path, 'rb') as f:
            metadata, images = pickle.load(f)
        
        all_images.append(images[:, 0:1, :, :])  # First channel only
        all_mass.append(metadata["mass"])
        all_label.append(metadata["label"])
    
    all_images = torchify(np.concatenate(all_images, axis=0))
    all_mass = torchify(np.concatenate(all_mass, axis=0))
    all_label = torchify(np.concatenate(all_label, axis=0))
    
    # Normalize images to [0, 1]
    img_min, img_max = all_images.min(), all_images.max()
    all_images = (all_images - img_min) / (img_max - img_min + 1e-8)
    
    print(f"Images: {all_images.shape}, range: [{all_images.min():.3f}, {all_images.max():.3f}]")
    print(f"Mass: range [{all_mass.min():.2f}, {all_mass.max():.2f}]")
    print(f"Labels: {torch.unique(all_label).tolist()}")
    
    return all_images, {"mass": all_mass, "label": all_label}


def collate_fn_fm(batch: list[tuple], fm_model, device):
    """Collate function - generates FM embeddings on the fly."""
    images, conditions = [], []
    for image, condition in batch:
        images.append(image)
        conditions.append(condition)
    
    images = torch.stack(images).to(device)
    conditions = torch.stack(conditions).to(device)
    embeddings = fm_model.sample_flow(conditions)
    
    return images, embeddings, conditions


def load_datasets(model_path: str, device: torch.device, batch_size: int = 32):
    """Load datasets and FM model."""
    fm_model = load_fm_model(checkpoint_path=model_path, device=device)
    fm_model.eval()
    
    all_images, features = load_images_with_features()
    conditions = torch.stack([features[name] for name in LABEL_NAMES], dim=1)
    
    # Train/val split
    train_size = int(0.8 * len(all_images))
    generator = torch.Generator().manual_seed(42)
    dataset = TensorDataset(all_images, conditions)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size], generator=generator
    )
    
    # Separate FM models for train/val to avoid embedding leakage
    val_fm_model = copy.deepcopy(fm_model)
    val_fm_model.eval()
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=partial(collate_fn_fm, fm_model=fm_model, device=device)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_fn_fm, fm_model=val_fm_model, device=device)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader, fm_model, val_dataset


# =============================================================================
# TRAINING
# =============================================================================

def training_script(output_dir: str, weights_path: str,
                    epochs: int = 250, batch_size: int = 32,
                    resume: str = None, auto_resume: bool = False):
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training FM-DDPM on device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------
    # 1) Load data
    # -------------------------------------------------
    train_loader, val_loader, fm_model, val_dataset = load_datasets(weights_path, device, batch_size)
    
    # -------------------------------------------------
    # 2) Setup fixed evaluation set
    # -------------------------------------------------
    eval_size = min(128, len(val_dataset))
    eval_indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(123))[:eval_size]
    
    eval_images = torch.stack([val_dataset[i][0] for i in eval_indices]).to(device)
    eval_conditions = torch.stack([val_dataset[i][1] for i in eval_indices]).to(device)
    eval_masses = eval_conditions[:, 0]
    eval_labels = eval_conditions[:, 1]
    
    # Generate fixed embeddings for evaluation
    with torch.no_grad():
        eval_cond = fm_model.sample_flow(eval_conditions)
    
    # -------------------------------------------------
    # 3) Create DDPM
    # -------------------------------------------------
    print("Creating DDPM model...")
    diffusion_model = DDPM(patch_size=4, schedule="cosine").to(device)
    ema_model = copy.deepcopy(diffusion_model)
    ema_decay = 0.9999
    
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=3e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    best_flux_corr = -1.0
    cfg_dropout = 0.1
    eval_every = 5
    start_epoch = 0
    
    # -------------------------------------------------
    # Resume
    # -------------------------------------------------
    if resume is None and auto_resume:
        auto_path = os.path.join(output_dir, "latest_checkpoint.pt")
        if os.path.exists(auto_path):
            print(f"Auto-resuming from {auto_path}")
            resume = auto_path
    
    wandb_id = None
    if resume and os.path.exists(resume):
        print(f"Loading checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            diffusion_model.load_state_dict(ckpt['model_state_dict'])
            ema_model.load_state_dict(ckpt['ema_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_flux_corr = ckpt.get('best_flux_corr', -1.0)
            best_val_loss = ckpt.get('best_val_loss', float("inf"))
            wandb_id = ckpt.get('wandb_run_id', None)
            print(f"  Resumed from epoch {ckpt['epoch']}")
        else:
            ema_model.load_state_dict(ckpt)
            diffusion_model.load_state_dict(ckpt)
            print(f"  Loaded weights only")
    
    # -------------------------------------------------
    # Wandb
    # -------------------------------------------------
    run = wandb.init(
        entity="matttsss-epfl",
        project="astropt_diffusion",
        name="Diffusion-FM",
        id=wandb_id,
        resume="must" if wandb_id else None,
        config={"epochs": epochs, "batch_size": batch_size}
    )
    
    def save_checkpoint(epoch, is_best_loss=False, is_best_corr=False, is_periodic=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': diffusion_model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_flux_corr': best_flux_corr,
            'best_val_loss': best_val_loss,
            'wandb_run_id': run.id if run else None,
        }
        torch.save(ckpt, os.path.join(output_dir, "latest_checkpoint.pt"))
        
        if is_best_loss:
            torch.save(ema_model.state_dict(), os.path.join(output_dir, "best_loss_model.pt"))
        if is_best_corr:
            torch.save(ckpt, os.path.join(output_dir, "best_corr_checkpoint.pt"))
            torch.save(ema_model.state_dict(), os.path.join(output_dir, "best_corr_model.pt"))
        if is_periodic:
            torch.save(ckpt, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    # -------------------------------------------------
    # 4) Training loop
    # -------------------------------------------------
    print("Starting training...")
    
    for epoch in range(start_epoch, epochs):
        diffusion_model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, conditions, _ in pbar:
            # CFG dropout
            if torch.rand(1).item() < cfg_dropout:
                conditions = torch.zeros_like(conditions)
            
            B = images.size(0)
            t = torch.randint(0, diffusion_model.timesteps, (B,), dtype=torch.long, device=device)
            
            noise = torch.randn_like(images)
            x_t = diffusion_model.q_sample(images, t, noise)
            noise_pred = diffusion_model.eps_model(x_t, t, conditions)
            
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), diffusion_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        lr_scheduler.step()
        
        # Validation
        ema_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, conditions, _ in val_loader:
                B = images.size(0)
                t = torch.randint(0, ema_model.timesteps, (B,), dtype=torch.long, device=device)
                
                noise = torch.randn_like(images)
                x_t = ema_model.q_sample(images, t, noise)
                noise_pred = ema_model.eps_model(x_t, t, conditions)
                
                val_loss += F.mse_loss(noise_pred, noise).item()
        
        val_loss /= len(val_loader)
        
        # Logging
        log_dict = {
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        
        # Full evaluation
        if epoch % eval_every == 0:
            print(f"\n  Evaluating at epoch {epoch}...")
            
            with torch.no_grad():
                generated = ema_model.ddim_sample(eval_cond, steps=50, eta=0.0, guidance_scale=2.0)
                generated = (generated - generated.min()) / (generated.max() - generated.min() + 1e-8)
                
                metrics = compute_all_metrics(eval_images, generated, eval_masses, eval_labels)
                
                for k, v in metrics.items():
                    if not k.startswith("_") and isinstance(v, (int, float)):
                        log_dict[k] = v
                
                # Plots
                fig_psd = plot_psd_comparison(metrics, title=f"Epoch {epoch}")
                log_dict["plots/psd"] = wandb.Image(fig_psd)
                plt.close(fig_psd)
                
                fig_samples = plot_generated_samples(generated, eval_masses, eval_labels)
                log_dict["plots/samples"] = wandb.Image(fig_samples)
                plt.close(fig_samples)
                
                fig_cond = plot_conditioning_quality(generated, eval_masses, eval_labels, eval_images)
                log_dict["plots/conditioning"] = wandb.Image(fig_cond)
                plt.close(fig_cond)
                
                fig_dist = plot_distribution_comparison(eval_images, generated)
                log_dict["plots/distributions"] = wandb.Image(fig_dist)
                plt.close(fig_dist)
        
        run.log(log_dict)
        
        # Print
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}", end="")
        if epoch % eval_every == 0:
            flux_corr = metrics.get("cond/flux_mass_corr", 0)
            mass_r2 = metrics.get("cond/mass_r2", 0)
            psd_ratio = metrics.get("psd/psd_ratio_mean", 0)
            print(f" | flux_corr={flux_corr:.3f}, mass_r2={mass_r2:.3f}, psd_ratio={psd_ratio:.3f}")
            
            if flux_corr > best_flux_corr:
                best_flux_corr = flux_corr
                print(f"  ★ New best flux_corr={flux_corr:.4f}")
                save_checkpoint(epoch, is_best_corr=True)
        else:
            print()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, is_best_loss=True)
        
        if epoch % 50 == 0 and epoch > 0:
            save_checkpoint(epoch, is_periodic=True)
        
        if epoch % 10 == 0:
            save_checkpoint(epoch)
    
    torch.save(ema_model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    run.finish()
    print(f"\nTraining complete! Best flux_corr={best_flux_corr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="model_weights/bestdifffm.pt")
    parser.add_argument("--weights_path", type=str, default="model_weights/fm.pt")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    args = parser.parse_args()
    
    training_script(
        args.output_dir,
        args.weights_path,
        args.epochs,
        args.batch_size,
        args.resume,
        args.auto_resume,
    )
