# scripts/diffusion_train_v4.py
#
# Training script with:
# - Resumption support (compatible with old checkpoint format)
# - Full metrics suite (r(ℓ), PSD, conditioning fidelity, peaks, etc.)
# - [0, 1] normalization (matching your working training)

import copy
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functools import partial

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from scripts.model_utils import compute_embeddings, load_astropt_model, load_fm_model
from scripts.embedings_utils import merge_datasets
from generative_model.DDPM import DDPM

LABEL_NAMES = ["mass", "label"]
DATASETS = [
        "data/BAHAMAS/bahamas_0.1.pkl",
        "data/BAHAMAS/bahamas_0.3.pkl",
        "data/BAHAMAS/bahamas_1.pkl",
        "data/BAHAMAS/bahamas_cdm.pkl",
    ]


# =============================================================================
# FOURIER-SPACE METRICS
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


def compute_cross_power_spectrum(image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-power spectrum between two images."""
    fft1 = np.fft.fftshift(np.fft.fft2(image1))
    fft2 = np.fft.fftshift(np.fft.fft2(image2))
    
    cross_power = np.real(fft1 * np.conj(fft2))
    
    H, W = image1.shape
    center = H // 2
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    n_bins = H // 2
    ell_edges = np.linspace(0, n_bins, n_bins + 1)
    ell_centers = (ell_edges[:-1] + ell_edges[1:]) / 2
    
    P_cross = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= ell_edges[i]) & (r < ell_edges[i + 1])
        if mask.sum() > 0:
            P_cross[i] = cross_power[mask].mean()
    
    return ell_centers, P_cross


def compute_cross_correlation_r_ell(truth: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation coefficient r(ℓ) - THE key metric from weak lensing."""
    ell, P_truth = compute_power_spectrum_2d(truth)
    _, P_recon = compute_power_spectrum_2d(reconstruction)
    _, P_cross = compute_cross_power_spectrum(truth, reconstruction)
    
    denominator = np.sqrt(P_truth * P_recon)
    denominator = np.maximum(denominator, 1e-10)
    
    r_ell = P_cross / denominator
    r_ell = np.clip(r_ell, -1, 1)
    
    return ell, r_ell


def compute_fourier_metrics(real_images: torch.Tensor, 
                            generated_images: torch.Tensor) -> dict:
    """Compute all Fourier-space metrics."""
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    
    if not np.isfinite(gen_np).all():
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    if real_np.ndim == 4:
        real_np = real_np[:, 0]
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    B = real_np.shape[0]
    
    all_r_ell = []
    all_psd_ratio = []
    ell = None
    
    for i in range(B):
        try:
            ell_i, r_ell = compute_cross_correlation_r_ell(real_np[i], gen_np[i])
            if ell is None:
                ell = ell_i
            
            if np.isfinite(r_ell).all():
                all_r_ell.append(r_ell)
            
            _, P_real = compute_power_spectrum_2d(real_np[i])
            _, P_gen = compute_power_spectrum_2d(gen_np[i])
            psd_ratio = np.log10((P_gen + 1e-10) / (P_real + 1e-10))
            
            if np.isfinite(psd_ratio).all():
                all_psd_ratio.append(psd_ratio)
        except Exception:
            continue
    
    if len(all_r_ell) == 0 or ell is None:
        n = 50
        return {
            "r_ell_low": 0.0, "r_ell_mid": 0.0, "r_ell_high": 0.0, "r_ell_mean": 0.0,
            "psd_ratio_low": 0.0, "psd_ratio_mid": 0.0, "psd_ratio_high": 0.0, "psd_ratio_mae": 0.0,
            "_ell": np.arange(n), "_r_ell": np.zeros(n), "_psd_ratio": np.zeros(n),
        }
    
    mean_r_ell = np.mean(all_r_ell, axis=0)
    mean_psd_ratio = np.mean(all_psd_ratio, axis=0) if all_psd_ratio else np.zeros_like(mean_r_ell)
    
    n = len(ell)
    low = slice(1, n // 3)
    mid = slice(n // 3, 2 * n // 3)
    high = slice(2 * n // 3, n)
    
    return {
        "r_ell_low": float(mean_r_ell[low].mean()),
        "r_ell_mid": float(mean_r_ell[mid].mean()),
        "r_ell_high": float(mean_r_ell[high].mean()),
        "r_ell_mean": float(mean_r_ell[1:].mean()),
        "psd_ratio_low": float(mean_psd_ratio[low].mean()),
        "psd_ratio_mid": float(mean_psd_ratio[mid].mean()),
        "psd_ratio_high": float(mean_psd_ratio[high].mean()),
        "psd_ratio_mae": float(np.abs(mean_psd_ratio[1:]).mean()),
        "_ell": ell, "_r_ell": mean_r_ell, "_psd_ratio": mean_psd_ratio,
    }


# =============================================================================
# PIXEL-SPACE METRICS
# =============================================================================

def compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 7) -> float:
    """Structural Similarity Index."""
    from scipy.ndimage import uniform_filter
    
    C1, C2 = 0.01**2, 0.03**2
    
    mu1 = uniform_filter(img1, size=window_size)
    mu2 = uniform_filter(img2, size=window_size)
    
    sigma1_sq = uniform_filter(img1**2, size=window_size) - mu1**2
    sigma2_sq = uniform_filter(img2**2, size=window_size) - mu2**2
    sigma12 = uniform_filter(img1 * img2, size=window_size) - mu1 * mu2
    
    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_pixel_metrics(real_images: torch.Tensor, generated_images: torch.Tensor) -> dict:
    """Pixel-level reconstruction metrics."""
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    
    if not np.isfinite(gen_np).all():
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    if real_np.ndim == 4:
        real_np = real_np[:, 0]
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    B = real_np.shape[0]
    pearson_list, ssim_list, mse_list = [], [], []
    
    for i in range(B):
        r_flat, g_flat = real_np[i].flatten(), gen_np[i].flatten()
        
        if r_flat.std() > 1e-10 and g_flat.std() > 1e-10:
            corr, _ = pearsonr(r_flat, g_flat)
            if np.isfinite(corr):
                pearson_list.append(corr)
        
        ssim_val = compute_ssim(real_np[i], gen_np[i])
        if np.isfinite(ssim_val):
            ssim_list.append(ssim_val)
        
        mse_list.append(np.mean((real_np[i] - gen_np[i])**2))
    
    return {
        "pixel_pearson_r": float(np.mean(pearson_list)) if pearson_list else 0.0,
        "pixel_ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "pixel_mse": float(np.mean(mse_list)),
        "pixel_psnr": float(10 * np.log10(1.0 / (np.mean(mse_list) + 1e-10))),
    }


# =============================================================================
# PEAK STATISTICS
# =============================================================================

def compute_peak_metrics(real_images: torch.Tensor, generated_images: torch.Tensor) -> dict:
    """Peak (halo) detection metrics."""
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    
    if not np.isfinite(gen_np).all():
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    if real_np.ndim == 4:
        real_np = real_np[:, 0]
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    def count_peaks(img, snr_threshold=3):
        if img.std() < 1e-10:
            return 0
        threshold = img.mean() + snr_threshold * img.std()
        from scipy.ndimage import maximum_filter, label
        local_max = maximum_filter(img, size=3)
        peaks = (img == local_max) & (img > threshold)
        _, n_peaks = label(peaks)
        return n_peaks
    
    B = real_np.shape[0]
    real_peaks = [count_peaks(real_np[i]) for i in range(B)]
    gen_peaks = [count_peaks(gen_np[i]) for i in range(B)]
    
    mean_real = np.mean(real_peaks) if real_peaks else 0.0
    mean_gen = np.mean(gen_peaks) if gen_peaks else 0.0
    
    return {
        "peak_count_real": float(mean_real),
        "peak_count_gen": float(mean_gen),
        "peak_count_ratio": float(mean_gen / (mean_real + 1e-10)),
    }


# =============================================================================
# NON-GAUSSIAN STATISTICS
# =============================================================================

def compute_nongaussian_metrics(real_images: torch.Tensor, generated_images: torch.Tensor) -> dict:
    """Higher-order statistics."""
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    
    if not np.isfinite(gen_np).all():
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    if real_np.ndim == 4:
        real_np = real_np[:, 0]
    if gen_np.ndim == 4:
        gen_np = gen_np[:, 0]
    
    def safe_stat(func, arr):
        val = func(arr.flatten())
        return val if np.isfinite(val) else 0.0
    
    B = real_np.shape[0]
    real_skew = [safe_stat(skew, real_np[i]) for i in range(B)]
    gen_skew = [safe_stat(skew, gen_np[i]) for i in range(B)]
    real_kurt = [safe_stat(kurtosis, real_np[i]) for i in range(B)]
    gen_kurt = [safe_stat(kurtosis, gen_np[i]) for i in range(B)]
    
    return {
        "skewness_real": float(np.mean(real_skew)),
        "skewness_gen": float(np.mean(gen_skew)),
        "skewness_diff": float(np.abs(np.mean(gen_skew) - np.mean(real_skew))),
        "kurtosis_real": float(np.mean(real_kurt)),
        "kurtosis_gen": float(np.mean(gen_kurt)),
        "kurtosis_diff": float(np.abs(np.mean(gen_kurt) - np.mean(real_kurt))),
    }


# =============================================================================
# CONDITIONING FIDELITY
# =============================================================================

def extract_image_features(images: np.ndarray) -> np.ndarray:
    """Extract features for conditioning fidelity evaluation."""
    if images.ndim == 4:
        images = images[:, 0]
    
    B = images.shape[0]
    features = []
    
    for i in range(B):
        img = images[i]
        if not np.isfinite(img).all():
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        
        flux = img.sum()
        _, n_peaks_95 = ndimage.label(img > np.percentile(img, 95))
        _, n_peaks_99 = ndimage.label(img > np.percentile(img, 99))
        max_val = img.max()
        
        h, w = img.shape
        center_mask = np.zeros_like(img, dtype=bool)
        center_mask[h//4:3*h//4, w//4:3*w//4] = True
        concentration = img[center_mask].sum() / (img.sum() + 1e-10)
        
        bg_mask = img < np.percentile(img, 50)
        bg_std = img[bg_mask].std() if bg_mask.sum() > 1 else 0.0
        std = img.std() if img.std() > 0 else 1e-10
        
        feat = [flux, n_peaks_95, n_peaks_99, max_val, concentration, bg_std, std]
        feat = [0.0 if not np.isfinite(f) else f for f in feat]
        features.append(feat)
    
    return np.array(features)


def compute_conditioning_fidelity(real_images: torch.Tensor, generated_images: torch.Tensor,
                                   masses: torch.Tensor, labels: torch.Tensor) -> dict:
    """Can we recover mass/label from generated images?"""
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    masses_np = masses.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    if not np.isfinite(gen_np).all():
        gen_np = np.nan_to_num(gen_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    real_features = extract_image_features(real_np)
    gen_features = extract_image_features(gen_np)
    
    if not np.isfinite(real_features).all() or not np.isfinite(gen_features).all():
        real_features = np.nan_to_num(real_features, nan=0.0)
        gen_features = np.nan_to_num(gen_features, nan=0.0)
    
    metrics = {}
    
    # Mass recovery
    mass_regressor = Ridge(alpha=1.0)
    mass_regressor.fit(real_features, masses_np)
    
    pred_mass_real = mass_regressor.predict(real_features)
    pred_mass_gen = mass_regressor.predict(gen_features)
    
    r2_mass_real = r2_score(masses_np, pred_mass_real)
    r2_mass_gen = r2_score(masses_np, pred_mass_gen)
    
    metrics["mass_r2_real"] = float(r2_mass_real)
    metrics["mass_r2_gen"] = float(r2_mass_gen)
    metrics["mass_r2_retention"] = float(r2_mass_gen / (r2_mass_real + 1e-10))
    
    gen_flux = gen_np.sum(axis=(1, 2, 3)) if gen_np.ndim == 4 else gen_np.sum(axis=(1, 2))
    flux_corr, _ = pearsonr(gen_flux, masses_np)
    metrics["flux_mass_correlation"] = float(flux_corr) if np.isfinite(flux_corr) else 0.0
    
    # Label recovery
    if len(np.unique(labels_np)) > 1:
        label_regressor = Ridge(alpha=1.0)
        label_regressor.fit(real_features, labels_np)
        
        r2_label_real = r2_score(labels_np, label_regressor.predict(real_features))
        r2_label_gen = r2_score(labels_np, label_regressor.predict(gen_features))
        
        metrics["label_r2_real"] = float(r2_label_real)
        metrics["label_r2_gen"] = float(r2_label_gen)
        metrics["label_r2_retention"] = float(r2_label_gen / (r2_label_real + 1e-10))
    else:
        metrics["label_r2_retention"] = 1.0
    
    return metrics


# =============================================================================
# COMBINED METRICS
# =============================================================================

def compute_all_metrics(real_images: torch.Tensor, generated_images: torch.Tensor,
                        masses: torch.Tensor = None, labels: torch.Tensor = None) -> dict:
    """Compute all metrics."""
    metrics = {}
    
    fourier = compute_fourier_metrics(real_images, generated_images)
    metrics.update({f"fourier/{k}": v for k, v in fourier.items() if not k.startswith("_")})
    metrics["_fourier_raw"] = fourier
    
    pixel = compute_pixel_metrics(real_images, generated_images)
    metrics.update({f"pixel/{k}": v for k, v in pixel.items()})
    
    peak = compute_peak_metrics(real_images, generated_images)
    metrics.update({f"peak/{k}": v for k, v in peak.items()})
    
    nongauss = compute_nongaussian_metrics(real_images, generated_images)
    metrics.update({f"nongauss/{k}": v for k, v in nongauss.items()})
    
    if masses is not None and labels is not None:
        cond = compute_conditioning_fidelity(real_images, generated_images, masses, labels)
        metrics.update({f"conditioning/{k}": v for k, v in cond.items()})
    
    return metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_r_ell_curve(metrics: dict, title: str = "") -> plt.Figure:
    """Plot r(ℓ) curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    raw = metrics.get("_fourier_raw", metrics)
    ell, r_ell, psd_ratio = raw["_ell"], raw["_r_ell"], raw["_psd_ratio"]
    
    ax = axes[0]
    ax.semilogx(ell[1:], r_ell[1:], 'b-', linewidth=2)
    ax.axhline(1, color='gray', linestyle='--', label='Perfect')
    ax.axhline(0, color='gray', linestyle=':')
    ax.fill_between(ell[1:], 0.9, 1.0, alpha=0.2, color='green', label='>0.9')
    ax.set_xlabel("ℓ"); ax.set_ylabel("r(ℓ)"); ax.set_title(f"Cross-Correlation {title}")
    ax.set_ylim(-0.2, 1.1); ax.legend(loc='lower left'); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.semilogx(ell[1:], psd_ratio[1:], 'k-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.fill_between(ell[1:], -0.1, 0.1, alpha=0.2, color='green', label='±0.1 dex')
    ax.set_xlabel("ℓ"); ax.set_ylabel("log₁₀(P_gen/P_real)"); ax.set_title(f"PSD Ratio {title}")
    ax.set_ylim(-0.5, 0.5); ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sample_comparison(real_images: torch.Tensor, generated_images: torch.Tensor, n_samples: int = 4) -> plt.Figure:
    """Side-by-side comparison."""
    fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 10))
    
    for i in range(n_samples):
        real_img = real_images[i, 0].cpu().numpy()
        gen_img = generated_images[i, 0].cpu().numpy()
        diff_img = real_img - gen_img
        
        vmax = max(real_img.max(), gen_img.max())
        vmin = min(real_img.min(), gen_img.min())
        
        axes[0, i].imshow(real_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"Real #{i}"); axes[0, i].axis('off')
        
        axes[1, i].imshow(gen_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"Generated #{i}"); axes[1, i].axis('off')
        
        axes[2, i].imshow(diff_img, cmap='RdBu', vmin=-np.abs(diff_img).max(), vmax=np.abs(diff_img).max())
        axes[2, i].set_title(f"Diff #{i}"); axes[2, i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_conditioning_scatter(real_images: torch.Tensor, generated_images: torch.Tensor, masses: torch.Tensor) -> plt.Figure:
    """Plot flux vs mass."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    masses_np = masses.cpu().numpy()
    
    if real_np.ndim == 4:
        real_np, gen_np = real_np[:, 0], gen_np[:, 0]
    
    real_flux = real_np.sum(axis=(1, 2))
    gen_flux = gen_np.sum(axis=(1, 2))
    
    corr_real, _ = pearsonr(masses_np, real_flux)
    corr_gen, _ = pearsonr(masses_np, gen_flux)
    
    axes[0].scatter(masses_np, real_flux, alpha=0.6, s=30)
    axes[0].set_xlabel("Mass"); axes[0].set_ylabel("Flux"); axes[0].set_title(f"Real (r={corr_real:.3f})")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(masses_np, gen_flux, alpha=0.6, s=30, color='orange')
    axes[1].set_xlabel("Mass"); axes[1].set_ylabel("Flux"); axes[1].set_title(f"Generated (r={corr_gen:.3f})")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# TRAINING SCRIPT
# =============================================================================
def load_images():
    dataset_images = merge_datasets(DATASETS, image_only=True)

    torchify = lambda column: torch.from_numpy(np.asarray(column)).float()

    # Pre-load images
    print("Pre-loading images...")
    all_images = torchify(dataset_images["image"])
    print(f"Raw images shape: {all_images.shape}")

    # Handle format - images are (N, C, H, W) already if shape[1] == 1
    if all_images.dim() == 4 and all_images.shape[1] != 1:
        all_images = all_images.permute(0, 3, 1, 2)
    elif all_images.dim() == 3:
        all_images = all_images.unsqueeze(1)

    # Normalize to [0, 1] (matching your working code)
    print(f"Raw images range: [{all_images.min():.4f}, {all_images.max():.4f}]")
    img_min, img_max = all_images.min(), all_images.max()
    all_images = (all_images - img_min) / (img_max - img_min + 1e-8)
    print(f"Normalized images range: [{all_images.min():.4f}, {all_images.max():.4f}]")
    
    print(f"Images shape: {all_images.shape}")


    return all_images, {feature_name: torchify(dataset_images[feature_name]) for feature_name in LABEL_NAMES}

def collate_fn(batch: list[tuple], fm_model, device):
    conditions = []
    images = []

    for image, condition in batch:
        conditions.append(condition)
        images.append(image)

    conditions = torch.stack(conditions).to(device)
    images = torch.stack(images).to(device)

    embeddings = fm_model.sample_flow(conditions)

    return images, embeddings

def load_fm_datasets(model_path: str, device: torch.device, batch_size: int = 32):

    fm_model = load_fm_model(checkpoint_path=model_path, device=device)
    all_images, features_dict = load_images()
    all_images = all_images.to(device)
    features = torch.stack([features_dict[feature_name].to(device) for feature_name in LABEL_NAMES], dim=1)

    # Split datasets
    # Train/val split with combined dataset
    train_size = int(0.8 * len(all_images))
  
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(features), generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = TensorDataset(all_images[train_indices], features[train_indices])
    val_dataset = TensorDataset(all_images[val_indices], features[val_indices])

    val_model = copy.deepcopy(fm_model)
    val_model.eval()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, device=device, fm_model=fm_model)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, device=device, fm_model=val_model)
    )

    # Fixed evaluation batch
    eval_size = min(128, len(val_dataset))
    eval_indices = val_indices[:eval_size]

    eval_images = all_images[eval_indices].to(device)
    eval_masses = features_dict['mass'][eval_indices].to(device)
    eval_labels = features_dict['label'][eval_indices].to(device)


    # Generate fixed embeddings for evaluation
    with torch.no_grad():
        eval_cond = fm_model.sample_flow(features[eval_indices])
    
    return train_loader, val_loader, (eval_images, eval_cond, eval_masses, eval_labels)


def load_astropt_datasets(model_path: str, device: torch.device, batch_size: int = 32):
    """
    Load AstroPT datasets, compute embeddings, and create train/val dataloaders.
    
    Args:
        model_path: Path to the finetuned AstroPT model checkpoint
        device: Device to use for computation (torch.device)
    
    Returns:
        Tuple of (train_loader, val_loader, lab_dict) where:
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - lab_dict: Dictionary with label information
    """
    
     # Load finetuned AstroPT encoder
    print("Loading finetuned AstroPT checkpoint...")
    model = load_astropt_model(checkpoint_path=model_path, device=device)
    model.eval()
    
    # Load datasets
    print("\nLoading datasets...")
    dataset = merge_datasets(DATASETS, feature_names=LABEL_NAMES)
    print(f"Dataset size: {len(dataset)}", end="\n\n")

    # Compute embeddings
    print("Computing embeddings...")
    dl = DataLoader(dataset, batch_size=512, num_workers=4, prefetch_factor=3)
    embeddings, features = compute_embeddings(model, dl, device, LABEL_NAMES)
    embeddings = embeddings.cpu()
    features = {k: v.cpu() for k, v in features.items()}

    all_images, _ = load_images()

    # Train/val split with combined dataset
    train_size = int(0.8 * len(embeddings))
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(embeddings), generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = TensorDataset(all_images[train_indices], embeddings[train_indices])
    val_dataset = TensorDataset(all_images[val_indices], embeddings[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Fixed evaluation batch
    eval_size = min(128, len(val_dataset))
    eval_indices = val_indices[:eval_size]

    eval_images = all_images[eval_indices].to(device)
    eval_cond = embeddings[eval_indices].to(device)
    eval_masses = features['mass'][eval_indices].to(device)
    eval_labels = features['label'][eval_indices].to(device)

    # Free memory
    del all_images, embeddings
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return train_loader, val_loader, (eval_images, eval_cond, eval_masses, eval_labels)


def training_script(output_dir: str, weights_path: str, timesteps: int = 1000,
                    epochs: int = 200, batch_size: int = 32, 
                    resume: str | None = None, auto_resume: bool = False, use_astropt: bool = True):
    """Training with resumption and full metrics."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Training diffusion on device: {device}")
    print(f"T={timesteps}, epochs={epochs}, batch_size={batch_size}", end="\n\n")

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------
    # 1) Load datasets
    # -------------------------------------------------
    if use_astropt:
        train_loader, val_loader, val_datas = load_astropt_datasets(weights_path, device)
    else:
        train_loader, val_loader, val_datas = load_fm_datasets(weights_path, device)

    eval_images, eval_cond, eval_masses, eval_labels = val_datas

    # -------------------------------------------------
    # 2) Create model, optimizer, EMA
    # -------------------------------------------------
    print("Creating DDPM model...")
    
    diffusion_model = DDPM(
        timesteps=timesteps,
        patch_size=4,
        schedule="cosine",
    ).to(device)
    
    ema_model = copy.deepcopy(diffusion_model)
    ema_decay = 0.9999

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=3e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_r_ell = -1.0
    cfg_dropout = 0.1
    eval_every = 5
    start_epoch = 0

    # -------------------------------------------------
    # 3) Resume from checkpoint if provided
    # -------------------------------------------------
    if resume is None and auto_resume:
        auto_path = os.path.join(output_dir, "latest_checkpoint.pt")
        if os.path.exists(auto_path):
            print(f"Auto-resuming from {auto_path}")
            resume = auto_path
    
    wandb_id = None
    if resume is not None and os.path.exists(resume):
        print(f"Loading checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        
        # Handle both old format (just state_dict) and new format (dict with keys)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_r_ell = checkpoint.get('best_r_ell', -1.0)
            best_val_loss = checkpoint.get('best_val_loss', float("inf"))
            wandb_id = checkpoint.get('wandb_run_id', None)
            print(f"  Resumed from epoch {checkpoint['epoch']}, best_r_ell={best_r_ell:.4f}")
        else:
            # Old format - just EMA state dict
            ema_model.load_state_dict(checkpoint)
            diffusion_model.load_state_dict(checkpoint)
            print(f"  Loaded old-format checkpoint (weights only)")
            # Can't resume optimizer/scheduler state, starting fresh from epoch 0
            # But we have the model weights

    # -------------------------------------------------
    # 4) Initialize Weights & Biases
    # -------------------------------------------------
    run = wandb.init(
        entity="matttsss-epfl",
        project="astropt_diffusion",
        name=f"Diffusion v4 - T={timesteps}",
        id=wandb_id,
        resume="must" if wandb_id else None,
        config={"timesteps": timesteps, "epochs": epochs, "batch_size": batch_size}
    )

    def save_checkpoint(epoch, is_best_loss=False, is_best_r_ell=False, is_periodic=False):
        """Save full training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': diffusion_model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_r_ell': best_r_ell,
            'best_val_loss': best_val_loss,
            'timesteps': timesteps,
            'wandb_run_id': run.id if run else None,
        }
        
        torch.save(checkpoint, os.path.join(output_dir, "latest_checkpoint.pt"))
        
        if is_best_loss:
            torch.save(ema_model.state_dict(), os.path.join(output_dir, "best_diffusion_model.pt"))
        
        if is_best_r_ell:
            torch.save(checkpoint, os.path.join(output_dir, "best_r_ell_checkpoint.pt"))
            torch.save(ema_model.state_dict(), os.path.join(output_dir, "best_r_ell_model.pt"))
        
        if is_periodic:
            torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))

    # -------------------------------------------------
    # 5) Training loop
    # -------------------------------------------------
    print("Starting training...")
    
    for epoch in range(start_epoch, epochs):
        diffusion_model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, conditions in pbar:
            images = images.to(device)
            conditions = conditions.to(device)
            
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
            pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
            for images, conditions in pbar:
                images = images.to(device)
                conditions = conditions.to(device)

                B = images.size(0)
                t = torch.randint(0, ema_model.timesteps, (B,), dtype=torch.long, device=device)

                noise = torch.randn_like(images)
                x_t = ema_model.q_sample(images, t, noise)
                noise_pred = ema_model.eps_model(x_t, t, conditions)

                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Logging
        log_dict = {
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        # Full metrics evaluation
        if epoch % eval_every == 0:
            print(f"\n  Running metrics at epoch {epoch}...")
            
            with torch.no_grad():
                generated = ema_model.ddim_sample(eval_cond, steps=50, eta=0.0, guidance_scale=2.0)
                
                # Normalize generated to [0, 1]
                generated = (generated - generated.min()) / (generated.max() - generated.min() + 1e-8)
                
                metrics = compute_all_metrics(eval_images, generated, eval_masses, eval_labels)
                
                for k, v in metrics.items():
                    if not k.startswith("_") and isinstance(v, (int, float)):
                        log_dict[k] = v
                
                # Plots
                fig_r_ell = plot_r_ell_curve(metrics, title=f"Epoch {epoch}")
                log_dict["plots/r_ell"] = wandb.Image(fig_r_ell)
                plt.close(fig_r_ell)
                
                fig_samples = plot_sample_comparison(eval_images, generated, n_samples=4)
                log_dict["plots/samples"] = wandb.Image(fig_samples)
                plt.close(fig_samples)
                
                fig_cond = plot_conditioning_scatter(eval_images, generated, eval_masses)
                log_dict["plots/conditioning"] = wandb.Image(fig_cond)
                plt.close(fig_cond)

            r_ell_mean = metrics.get("fourier/r_ell_mean", 0)
            r_ell_low = metrics.get("fourier/r_ell_low", 0)
            mass_ret = metrics.get("conditioning/mass_r2_retention", 0)
            print(f" | r(ℓ)_mean={r_ell_mean:.3f}, r(ℓ)_low={r_ell_low:.3f}, mass_ret={mass_ret:.3f}")
            
            if r_ell_mean > best_r_ell:
                best_r_ell = r_ell_mean
                print(f"  ★ New best r(ℓ)={r_ell_mean:.4f}")
                save_checkpoint(epoch, is_best_r_ell=True)

        else:
            print()

        run.log(log_dict)
        # Print summary
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}", end="")

        # Save best by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, is_best_loss=True)

        # Periodic save
        if epoch % 50 == 0 and epoch > 0:
            save_checkpoint(epoch, is_periodic=True)
        
        # Always save latest every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(epoch)

    # Save final model
    torch.save(ema_model.state_dict(), os.path.join(output_dir, f"final_diffusion_model_{timesteps}.pt"))
    
    run.finish()
    print(f"\nTraining complete! Best r(ℓ)={best_r_ell:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="model_weights/")
    parser.add_argument("--weights_path", type=str, default="model_weights/finetuned_contrastive_ckpt.pt")
    parser.add_argument("--model_type", type=str, choices=["astropt", "fm"], default="astropt")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--auto_resume", action="store_true", help="Auto-resume from latest")
    args = parser.parse_args()

    training_script(
        args.output_dir, 
        args.weights_path, 
        args.timesteps,
        args.epochs,
        args.batch_size,
        args.resume,
        args.auto_resume,
        args.model_type == "astropt"
    )
