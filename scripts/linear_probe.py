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
    
    parser.add_argument('--reset', action='store_true', default=False, help='Whether to reset cached embeddings')
    args = parser.parse_args()

    def train_probe(zs, ys):
        probe = LinearRegression()
        probe.fit(zs, ys)
        return probe
    
    print("Loading/generating embeddings...")
    zss, yss = get_embeddings(reset=args.reset)

    print("Training probe...")
    # Now we train a linear probe on half the data and test on the other half
    # In a "real" setting you may want to use a more powerful model than a linear regressor
    # (and possibly a more difficult problem then magnitude prediction ;) !)
    halfway = len(zss) // 2
    probe = train_probe(zss[:halfway], yss[:halfway])
    pss = probe.predict(zss[halfway:])
    print(
        f"MSE: {mean_squared_error(pss, yss[halfway:])} R2: {r2_score(pss, yss[halfway:])}"
    )

    # Now let's visualise the embedding space by performing UMAP and plotting
    umap = UMAP(n_components=2)
    zss = StandardScaler().fit_transform(zss)
    X_umap = umap.fit_transform(zss)

    # Plot ground truth magnitude with PCA components
    vmax = np.percentile(yss, 95)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=yss, vmax=vmax, cmap="viridis")
    plt.colorbar(label="mag_g")
    plt.savefig("figures/pca_mag_g.png", dpi=300)
    plt.show()
    plt.clf()

    # Plot predicted vs ground truth magnitude
    reg = LinearRegression().fit(yss[halfway:].reshape(-1, 1), pss)
    plt.plot(yss[halfway:], pss, ".")
    plt.plot(
        yss[halfway:],
        reg.predict(yss[halfway:].reshape(-1, 1)),
        "--",
        color="red",
        label=f"Linear fit: $pred = {reg.coef_[0]:.2f} \\cdot x + {reg.intercept_:.2f}$",
    )
    plt.xlabel("Ground truth magnitude")
    plt.ylabel("Predicted magnitude")
    plt.legend()
    plt.savefig("figures/predicted_vs_ground_truth.png", dpi=300)
    plt.show()
