"""
Plotting Utilities Module

This module provides common plotting utilities for visualization of embeddings and model
predictions. Includes UMAP projections, linear probe predictions, cross-section histograms,
and customizable figure layouts with standardized font sizes.

Key Features:
    - Configurable matplotlib font sizes and styles
    - Generic grid-based plotting for multiple labels
    - Cross-section histogram visualization
    - UMAP projections with linear probe evaluation
    - Automated figure saving and display

"""

import matplotlib.pyplot as plt
import numpy as np

from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def set_fonts(**font_sizes):
    """
    Configure matplotlib font sizes for consistent plotting appearance.
    
    Sets default font sizes for all matplotlib elements including axes labels,
    tick labels, legend, and figure titles.
    
    Args:
        **font_sizes: Keyword arguments to override default sizes:
            - font (int): Default font size (default: 16)
            - axes_title (int): Axes title size (default: 18)
            - axes_label (int): X/Y axes label size (default: 18)
            - xtick (int): X-axis tick label size (default: 18)
            - ytick (int): Y-axis tick label size (default: 18)
            - legend (int): Legend font size (default: 18)
            - figure_title (int): Figure title size (default: 20)
    
    Returns:
        None: Modifies matplotlib rcParams globally
    """
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=font_sizes.get('font', SMALL_SIZE))                 # controls default text sizes
    plt.rc('axes', titlesize=font_sizes.get('axes_title', MEDIUM_SIZE))     # fontsize of the axes title
    plt.rc('axes', labelsize=font_sizes.get('axes_label', MEDIUM_SIZE))     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_sizes.get('xtick', MEDIUM_SIZE))         # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_sizes.get('ytick', MEDIUM_SIZE))         # fontsize of the tick labels
    plt.rc('legend', fontsize=font_sizes.get('legend', MEDIUM_SIZE))        # legend fontsize
    plt.rc('figure', titlesize=font_sizes.get('figure_title', BIGGER_SIZE)) # fontsize of the figure title


def plot_labels(plot_func, title, label_names: list[str], **kwargs):
    """
    Generic function to create a grid of plots using a provided plotting function.
    
    Arranges multiple subplots in a grid (maximum 4 per row) and applies the provided
    plotting function to each subplot, useful for visualizing multiple labels in a
    consistent layout.
    
    Args:
        plot_func: Callable with signature plot_func(fig, ax, label_name, **kwargs)
                   that plots on the given axis for a specific label
        title (str): Overall figure title
        label_names (list[str]): List of label names to create subplots for
        **kwargs: Additional keyword arguments passed to plot_func for each label
    
    Returns:
        plt.Figure: The created figure object
    """

    # maximum 4 plots per row
    nplots = len(label_names)
    ncols = min(4, nplots) if nplots > 0 else 1
    nrows = (nplots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    
    # Ensure axs is a 1D array we can index into
    axs = np.atleast_1d(axs).reshape(-1)
    for ax, label_name in zip(axs, label_names):
        plot_func(fig, ax, label_name, **kwargs)
    
    # hide any unused axes (when nplots < nrows * ncols)
    for j in range(nplots, axs.size):
        axs[j].axis('off')

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_cross_section_histogram(ax, ref_values, pred_values, pred_method_name, bin_range=(-0.1, 1.1), nbins=30):
    """
    Plot histograms of predictions grouped by reference value classes.
    
    Creates overlaid histograms for predicted values, with each histogram corresponding to
    a unique reference value (cross-section). Computes and displays mean and standard
    deviation for each class.
    
    Args:
        ax (matplotlib.axes.Axes): Target axis for plotting
        ref_values (array-like): Reference/ground truth values for grouping, shape (num_samples,)
        pred_values (array-like): Predicted values to histogram, shape (num_samples,)
        pred_method_name (str): Name of prediction method for plot title
        bin_range (tuple): Histogram range as (min, max) (default: (-0.1, 1.1))
        nbins (int): Number of histogram bins (default: 30)
    
    Returns:
        None: Modifies ax in place
    """
    unique_values = np.unique(ref_values)
    alpha = 1 / len(unique_values)
    for label in unique_values:
        mask = (ref_values == label)

        mean_lin_pred = pred_values[mask].mean()
        std_lin_pred = pred_values[mask].std()
        ax.hist(pred_values[mask], alpha=alpha, bins=nbins,
                range=bin_range, label=f'$\\sigma_{{{label:.2f}}}$: {mean_lin_pred:.2f} ± {std_lin_pred:.2f}')

    ax.set_title(pred_method_name)
    ax.set_xlabel("Ground truth $\\sigma$")
    ax.set_ylabel("Predicted $\\sigma$")
    ax.legend()


def plot_probes(embeddings, labels_dict, data_name):
    """
    Visualize embeddings with UMAP and train linear probes for label prediction.
    
    Creates two sets of visualizations:
    1. UMAP projections of embeddings colored by each label
    2. Linear probe predictions (normalized) vs ground truth for each label
    
    Automatically saves figures to 'figures/' directory.
    
    Args:
        embeddings (np.ndarray): High-dimensional embeddings of shape (num_samples, embedding_dim)
        labels_dict (dict): Dictionary mapping label names to label arrays
        data_name (str): Dataset name for plot titles and filenames
    
    Returns:
        None: Displays plots and saves figures to disk
    
    Saved Files:
        - figures/umap_{data_name}_magnitudes.png: UMAP projections colored by labels
        - figures/{data_name}_pred.png: Linear probe predictions with metrics
    """

    ## Start with UMAP projection
    umap = UMAP(n_components=2)
    umap_embeddings = StandardScaler().fit_transform(embeddings)
    umap_embeddings = umap.fit_transform(umap_embeddings)


    # =============================================================================
    # Plot UMAP projections colored by each label
    # =============================================================================

    def plot_probe(fig, ax, label_name, umap_embeddings, labels_dict):
        data = labels_dict[label_name]
        vmax = np.percentile(data, 95)
        sc = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=data, vmax=vmax, cmap="viridis")

        ax.set_title(label_name)
        fig.colorbar(sc, ax=ax, label=label_name)

    fig = plot_labels(plot_probe, f"UMAP projection of {data_name} embeddings", list(labels_dict.keys()),
                      umap_embeddings=umap_embeddings, labels_dict=labels_dict)

    fig.savefig(f"figures/umap_{data_name}_magnitudes.png", dpi=300)
    plt.show()
    plt.clf()


    # =============================================================================
    # Plot Probe predictions
    # =============================================================================

    def plot_probe_predictions(fig, ax, label_name, embeddings, labels_dict):
        label_data = labels_dict[label_name]

        # Normalize label data
        mean, std = label_data.mean(), label_data.std()
        label_data = (label_data - mean) / std

        # Train linear probe to predict label from embeddings
        halfway = len(embeddings) // 2
        probe = LinearRegression().fit(embeddings[:halfway], label_data[:halfway])
        pss = probe.predict(embeddings[halfway:])

        # Compute metrics
        mse = mean_squared_error(pss, label_data[halfway:])
        r2 = r2_score(pss, label_data[halfway:])

        # Plot predictions vs ground truth
        reg = LinearRegression().fit(label_data[halfway:].reshape(-1, 1), pss)
        ax.scatter(label_data[halfway:], pss, alpha=0.1)
        ax.plot(
            label_data[halfway:],
            reg.predict(label_data[halfway:].reshape(-1, 1)),
            color="red",
            label=f"$pred = {reg.coef_[0]:.2f} \\cdot x + {reg.intercept_:.2f}$",
        )
        
        ax.set_title(f"Predictions for {label_name}\nMSE: {mse:.4f}, R2: {r2:.4f}")
        ax.set_xlabel(f"Ground truth {label_name}")
        ax.set_ylabel(f"Predicted {label_name}")
        ax.legend()
    
    fig = plot_labels(plot_probe_predictions, f"Normalized linear probe predictions of {data_name} embeddings", 
                      list(labels_dict.keys()), embeddings=embeddings, labels_dict=labels_dict)
    
    fig.savefig(f"figures/{data_name}_pred.png", dpi=300)
    plt.show()
    plt.clf()
