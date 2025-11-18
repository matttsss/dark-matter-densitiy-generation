"""
Example script that samples from a trained astropt model and finetunes
embeddings on a linear probe
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from embedings_utils import get_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PlotstroPT',
                    description='Generates plots for linear probe on astroPT embeddings')
    
    parser.add_argument('--astro_pt_data', action='store_true', default=False, help='Whether to use astroPT data for embeddings (default: False, dark data)')
    parser.add_argument('--nb_points', type=int, default=1000, help='Number of points to use for embeddings')
    parser.add_argument('--labels', nargs='+', default=["mass"], help='Labels to use for embeddings')
    args = parser.parse_args()
    
    labels = args.labels
    
    print("Loading/generating embeddings...")
    zss, labels = get_embeddings(args.astro_pt_data, args.nb_points, *labels)

    # Now let's visualise the embedding space by performing UMAP and plotting
    umap = UMAP(n_components=2)
    zss = StandardScaler().fit_transform(zss)
    X_umap = umap.fit_transform(zss)

    # Plot ground truth magnitude with UMAP components
    nplots = len(labels)
    # maximum 4 plots per row
    ncols = min(4, nplots) if nplots > 0 else 1
    nrows = (nplots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    
    # Ensure axs is a 1D array we can index into
    axs = np.atleast_1d(axs).reshape(-1)
    for i, (label, data) in enumerate(labels.items()):
        vmax = np.percentile(data, 95)
        sc = axs[i].scatter(X_umap[:, 0], X_umap[:, 1], c=data, vmax=vmax, cmap="viridis")

        axs[i].set_title(label)
        fig.colorbar(sc, ax=axs[i], label=label)
    
    # hide any unused axes (when nplots < nrows * ncols)
    for j in range(nplots, axs.size):
        axs[j].axis('off')

    fig.suptitle(f"UMAP projection of {'AstroPT galaxy embeddings' if args.astro_pt_data else 'Dark Matter embeddings'}")
    fig.tight_layout()

    if args.astro_pt_data:
        fig.savefig("figures/umap_astropt_magnitudes.png", dpi=300)
    else:
        fig.savefig("figures/umap_dark_magnitudes.png", dpi=300)
    
    plt.show()
    plt.clf()

    # Now let's train a linear probe to predict magnitude from embeddings
    def train_probe(zs, ys):
        probe = LinearRegression()
        probe.fit(zs, ys)
        return probe

    nplots = len(labels)
    # maximum 4 plots per row
    ncols = min(4, nplots) if nplots > 0 else 1
    nrows = (nplots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    
    # Ensure axs is a 1D array we can index into
    axs = np.atleast_1d(axs).reshape(-1)

    for i, (label, data) in enumerate(labels.items()):        
        print(f"Training probe on label {label}...")
        # Now we train a linear probe on half the data and test on the other half
        # In a "real" setting you may want to use a more powerful model than a linear regressor
        # (and possibly a more difficult problem then magnitude prediction ;) !)
        halfway = len(zss) // 2
        probe = train_probe(zss[:halfway], data[:halfway])
        pss = probe.predict(zss[halfway:])
        print(
            f"MSE: {mean_squared_error(pss, data[halfway:])} R2: {r2_score(pss, data[halfway:])}"
        )

        # Plot predicted vs ground truth magnitude
        reg = LinearRegression().fit(data[halfway:].reshape(-1, 1), pss)
        axs[i].scatter(data[halfway:], pss)
        axs[i].plot(
            data[halfway:],
            reg.predict(data[halfway:].reshape(-1, 1)),
            "--",
            color="red",
            label=f"Linear fit: $pred = {reg.coef_[0]:.2f} \\cdot x + {reg.intercept_:.2f}$",
        )
        
        axs[i].set_xlabel(f"Ground truth {label}")
        axs[i].set_ylabel(f"Predicted {label}")
        axs[i].legend()

    # hide any unused axes (when nplots < nrows * ncols)
    for j in range(nplots, axs.size):
        axs[j].axis('off')

    fig.suptitle(f"Linear probe predictions of {'AstroPT galaxy embeddings' if args.astro_pt_data else 'Dark Matter embeddings'}")
    fig.tight_layout()
    if args.astro_pt_data:
        fig.savefig("figures/astropt_pred.png", dpi=300)
    else:
        fig.savefig("figures/dark_pred.png", dpi=300)
    plt.show()
