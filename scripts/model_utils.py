"""
Model Utilities Module

This module provides utilities for loading, configuring, and working with AstroPT and
Flow Matching models. Includes functions for computing embeddings, loading checkpoints,
linear regression probes, and batch device management.

Key Features:
    - Efficient embedding computation from datasets
    - Model checkpoint loading with strict/flexible mode support
    - Linear regression layer for probe training
    - Running average metrics computation
    - Device-aware batch management

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from astropt.model import GPT, GPTConfig
from tqdm.auto import tqdm
from generative_model.vector_field import VectorField, VectorFieldConfig
from scripts.embeddings_utils import merge_datasets


@torch.no_grad()
def compute_embeddings(model, dataloader, device: torch.device, label_names: list[str],
                       disable_tqdm: bool = False):
    """
    Extract embeddings from a model over a full dataset.
    
    Iterates through dataloader, computes embeddings for each batch, and collects
    labels for the specified features. Returns stacked tensors of all embeddings
    and labels.
    
    Args:
        model: AstroPT model in evaluation mode (GPT)
        dataloader (DataLoader): DataLoader yielding batches with 'image' and label columns
        device (torch.device): Device for computation (cuda/mps/cpu)
        label_names (list[str]): Names of label columns to extract from batches
        disable_tqdm (bool): Whether to disable progress bar (default: False)
    
    Returns:
        tuple:
            - embeddings (torch.Tensor): Stacked embeddings of shape (num_samples, embedding_dim)
            - labels_dict (dict): Dictionary mapping label names to label tensors
    """
    model.eval()

    all_embeddings = []
    all_labels = {label: [] for label in label_names}

    for B in tqdm(dataloader, disable=disable_tqdm):
        B = batch_to_device(B, device)
        embeddings = model.generate_embeddings(B)["images"]
        all_embeddings.append(embeddings)

        for label in label_names:
            all_labels[label].append(B[label])

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = {label: torch.cat(all_labels[label], dim=0) for label in label_names}
    return all_embeddings, all_labels

def get_embeddings_datasets(model_path, device, label_names, split_ratio=0.8, nb_points=14000):
    """
    Load BAHAMAS datasets, compute embeddings, and split into train/validation sets.
    
    Loads BAHAMAS datasets, computes embeddings using provided model path,
    and splits into training and validation subsets.
    
    Args:
        model_path (str): Path to trained AstroPT model checkpoint
        device (torch.device): Device for computation (cuda/mps/cpu)
        label_names (list[str]): Names of conditions/labels to extract
        split_ratio (float): Fraction of data for training (default: 0.8)
        nb_points (int): Maximum number of samples to use (default: 14000)
    
    Returns:
        tuple:
            - embeddings_tuple: (train_embeddings, val_embeddings) tensors
            - conditions_tuple: (train_cond, val_cond) condition tensors of shape (..., num_conditions)
    """
    model = load_astropt_model(model_path, device=device, strict=True)
    dataset = merge_datasets([
        "data/BAHAMAS/bahamas_0.1.pkl", 
        "data/BAHAMAS/bahamas_0.3.pkl", 
        "data/BAHAMAS/bahamas_1.pkl",
        "data/BAHAMAS/bahamas_cdm.pkl"],
        feature_names=label_names, stack_features=False) \
            .shuffle(seed=42) \
            .take(nb_points)    

    has_metals = device.type == 'mps'
    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 512,
        num_workers = 0 if has_metals else 4,
        prefetch_factor = None if has_metals else 3
    )

    embeddings, cond_dict = compute_embeddings(model, dl, device, label_names)
    cond = torch.stack([cond_dict[k] for k in label_names], dim=-1)

    # Split into train and val
    nb_train = int(split_ratio * embeddings.size(0))

    train_embeddings = embeddings[:nb_train]
    val_embeddings = embeddings[nb_train:]
    train_cond = cond[:nb_train]
    val_cond = cond[nb_train:]

    return (train_embeddings, val_embeddings), (train_cond, val_cond)

class LinearRegression:
    """
    Simple linear regression layer for embedding-to-condition mapping.
    
    Implements least-squares linear regression. Used as
    a probe to evaluate embedding quality by mapping embeddings to conditions.
    """

    def __init__(self, device="cpu"):
        """
        Initialize linear regression layer.
        
        Args:
            device (str or torch.device): Device for tensors (default: "cpu")
        """
        super().__init__()
        self.device = device
        self.weights = None
        self.bias = None

    @staticmethod
    def _append_bias(X):
        return torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)

    @torch.no_grad()
    def fit(self, X, labels):
        """
        Fit linear regression using least-squares (pseudo-inverse method).
        
        Solves the normal equation: w = (X^T X)^-1 X^T y for weights and bias.
        
        Args:
            X (torch.Tensor or array-like): Input features of shape (num_samples, num_features)
            labels (torch.Tensor or array-like): Target labels of shape (num_samples,) or (num_samples, num_outputs)
        
        Returns:
            self: Returns self for method chaining
        """
        X = torch.as_tensor(X, device=self.device)
        labels = torch.as_tensor(labels, device=self.device)

        # Ensure labels is 2D (num_samples, num_outputs)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        if X.dim() == 1:
            X = X.unsqueeze(1)

        new_X = LinearRegression._append_bias(X)
        W = (torch.linalg.pinv(new_X.T @ new_X) @ new_X.T @ labels)

        self.weights = W[1:].T
        self.bias = W[0]

        return self

    def predict(self, X):
        """
        Predict labels for input features using fitted linear regression.
        
        Args:
            X (torch.Tensor or array-like): Input features of shape (num_samples, num_features)
        
        Returns:
            torch.Tensor: Predicted labels of shape (num_samples,) or (num_samples, num_outputs)
        
        Raises:
            ValueError: If model has not been fitted yet
        """
        X = torch.as_tensor(X, device=self.device)
        if self.weights is None: raise ValueError("Model has not been fitted yet.")
        return F.linear(X, self.weights, self.bias)


def load_fm_model(checkpoint_path, device, strict=True, **extra_model_config):
    """
    Load a Flow Matching model from checkpoint.
    
    Loads configuration from checkpoint and instantiates a VectorField model,
    then loads weights. Supports strict and flexible weight loading modes.
    
    Args:
        checkpoint_path (str): Path to model checkpoint containing 'config' and 'state_dict'
        device (torch.device or str): Device to load model onto (cuda/mps/cpu)
        strict (bool): If True, require exact weight match; if False, allow missing/extra keys (default: True)
        **extra_model_config: Additional config parameters to override checkpoint settings
    
    Returns:
        VectorField: Loaded flow matching model on the target device
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fm_config = VectorFieldConfig(**checkpoint["config"])

    for k, v in extra_model_config.items():
        setattr(fm_config, k, v)

    model = VectorField(fm_config)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    model.to(device)

    return model

def load_astropt_model(checkpoint_path, device, strict=True, **extra_model_config):
    """
    Load an AstroPT model from checkpoint.
    
    Loads model configuration, modality registry, and weights from checkpoint.
    Handles state dict prefix fixes and supports flexible weight loading.
    Can optionally retrieve target labels used for fine-tuning.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (torch.device or str): Device to load model onto (cuda/mps/cpu)
        strict (bool): If True, require exact weight match (default: True)
        **extra_model_config: Additional config parameters to override checkpoint settings
    
    Returns:
        GPT: Loaded AstroPT model
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_args = checkpoint["model_args"]
    modality_registry = checkpoint["modality_registry"]

    # Modify model for finetuning
    config = GPTConfig(**model_args)
    for k, v in extra_model_config.items():
        setattr(config, k, v)

    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model = GPT(config, modality_registry)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)

    return model

def batch_to_device(batch, device):
    """
    Recursively move batch data to target device.
    
    Handles mixed data types (tensors, dicts, lists, tuples) and moves all tensors
    to the specified device with non-blocking transfer for efficiency.
    
    Args:
        batch: Batch data (torch.Tensor, dict, list, tuple, or nested combinations)
        device (torch.device or str): Target device (cuda/mps/cpu)
    
    Returns:
        Same structure as input with all tensors moved to target device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list,tuple)):
        return type(batch)(batch_to_device(v, device) for v in batch)
    return batch

class RunningAverageMeter(object):
    """
    Compute running average and optionally store all values for later analysis.
    
    Maintains exponential moving average of metric values, useful for tracking
    training progress. Can optionally keep full history for plotting.
    """

    def __init__(self, momentum=0.99, keep_all=False):
        """
        Initialize running average meter.
        
        Args:
            momentum (float): Momentum for exponential moving average (default: 0.99)
            keep_all (bool): If True, store all values in losses list (default: False)
        """
        self.momentum = momentum
        self.losses = [] if keep_all else None
        self.reset()

    def reset(self):
        """Reset meter state (current value and average)."""
        self.val = None
        self.avg = 0
        self.losses = [] if self.losses is not None else None

    def update(self, val):
        """
        Update meter with new value.
        
        Args:
            val (float): New value to incorporate into running average
        """
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
    
    def register_loss(self, val):
        """
        Register value in loss history (if keep_all=True).
        
        Args:
            val (float): Loss value to add to history
        """
        if self.losses is not None:
            self.losses.append(val)