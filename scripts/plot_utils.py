import matplotlib.pyplot as plt
import numpy as np

from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def plot_labels(plot_func, embeddings, umap_embeddings, labels_dict, title):
    """Generic function to plot multiple labels in a grid using a provided plotting function
    
    Args:
        plot_func: function with signature (fig, ax, embeddings, umap_embeddings, label_name, label_data)
        embeddings: original high-dimensional embeddings
        umap_embeddings: 2D UMAP projections of embeddings
        labels_dict: dictionary of label name to label data
        title: overall figure title
    Returns:
        fig: matplotlib figure object
    """

    nplots = len(labels_dict)
    # maximum 4 plots per row
    ncols = min(4, nplots) if nplots > 0 else 1
    nrows = (nplots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    
    # Ensure axs is a 1D array we can index into
    axs = np.atleast_1d(axs).reshape(-1)
    for i, (label_name, data) in enumerate(labels_dict.items()):
        plot_func(fig, axs[i],  embeddings, umap_embeddings, label_name, data)
    
    # hide any unused axes (when nplots < nrows * ncols)
    for j in range(nplots, axs.size):
        axs[j].axis('off')

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_probes(embeddings, labels_dict, data_name):
    """
    Plots UMAP projections of embeddings colored by labels and trains linear probes
    to predict each label from the embeddings.

    Also saves the figures to files in `figures/`.

    Args:
        embeddings: High-dimensional embeddings
        labels_dict: Dictionary of label name to label data
        data_name: Name of the dataset (for titles and filenames)
    """

    ## Start with UMAP projection
    umap = UMAP(n_components=2)
    umap_embeddings = StandardScaler().fit_transform(embeddings)
    umap_embeddings = umap.fit_transform(umap_embeddings)


    # =============================================================================
    # Plot UMAP projections colored by each label
    # =============================================================================

    def plot_probe(fig, ax, embeddings, umap_embeddings, label_name, data):
        vmax = np.percentile(data, 95)
        sc = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=data, vmax=vmax, cmap="viridis")

        ax.set_title(label_name)
        fig.colorbar(sc, ax=ax, label=label_name)

    fig = plot_labels(plot_probe, embeddings, umap_embeddings, labels_dict,
                       f"UMAP projection of {data_name} embeddings")

    fig.savefig(f"figures/umap_{data_name}_magnitudes.png", dpi=300)
    plt.show()
    plt.clf()


    # =============================================================================
    # Plot Probe predictions
    # =============================================================================

    def plot_probe_predictions(fig, ax, embeddings, umap_embeddings, label_name, label_data):

        mean, std = label_data.mean(), label_data.std()
        label_data = (label_data - mean) / std

        halfway = len(embeddings) // 2
        probe = LinearRegression().fit(embeddings[:halfway], label_data[:halfway])
        pss = probe.predict(embeddings[halfway:])

        mse = mean_squared_error(pss, label_data[halfway:])
        r2 = r2_score(pss, label_data[halfway:])

        reg = LinearRegression().fit(label_data[halfway:].reshape(-1, 1), pss)
        ax.scatter(label_data[halfway:], pss)
        ax.plot(
            label_data[halfway:],
            reg.predict(label_data[halfway:].reshape(-1, 1)),
            "--",
            color="red",
            label=f"$pred = {reg.coef_[0]:.2f} \\cdot x + {reg.intercept_:.2f}$",
        )
        
        ax.set_title(f"Predictions for {label_name}\nMSE: {mse:.4f}, R2: {r2:.4f}")
        ax.set_xlabel(f"Ground truth {label_name}")
        ax.set_ylabel(f"Predicted {label_name}")
        ax.legend()

    fig = plot_labels(plot_probe_predictions, embeddings, umap_embeddings, labels_dict,
                       f"Normalized linear probe predictions of {data_name} embeddings")
    
    fig.savefig(f"figures/{data_name}_pred.png", dpi=300)
    plt.show()
    plt.clf()
