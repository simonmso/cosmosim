import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from Global import APS_COL_W as apsw


class MCMCResult:
    def __init__(
        self,
        chain,
        log_prob,
        labels=None,
        fiducial=None,
    ):
        self.chain = np.array(chain)
        self.log_prob = log_prob
        self.labels = labels
        self.fiducial = fiducial
        self.best = np.quantile(self.chain, 0.5, axis=1)

    def hist2d(self, q=[0.95, 0.68]):
        fig, ax = plt.subplots(figsize=(apsw, 0.9 * apsw))

        (_, _, _, img) = ax.hist2d(self.chain[0], self.chain[1], norm="log", bins=100)
        fig.colorbar(img, label="MCMC samples", location="bottom")

        # 1 and 2 sigma curves
        quantiles = -np.quantile(-self.log_prob, q)
        for i in range(len(q)):
            idx = self.log_prob > quantiles[i]

            included = self.chain[:, idx]

            hull = spatial.ConvexHull(included.T)

            for s in hull.simplices:
                ax.plot(included[0, s], included[1, s], c="white", ls="--")

        # Best value
        max_idx = np.argmax(self.log_prob)

        ax.scatter(
            self.chain[0, max_idx],
            self.chain[1, max_idx],
            marker="x",
            c="Red",
            label="Best",
            s=10,
        )

        if self.labels is not None:
            ax.set_xlabel(self.labels[0])
            ax.set_ylabel(self.labels[1])

        if self.fiducial is not None:
            ax.scatter(  # Include the fiducial values
                self.fiducial[0],
                self.fiducial[1],
                s=10,
                marker="*",
                c="k",
                label="Fiducial",
            )

        ax.set_title("MCMC Results")

        ax.legend(scatterpoints=1)

        return fig, ax

    def hist(self):
        q = [0.16, 0.50, 0.84]
        nplots = self.chain.shape[0]
        fig, axs = plt.subplots(nrows=nplots, figsize=(apsw, 0.5 * nplots * apsw))

        for i in range(nplots):
            quantiles = np.quantile(self.chain[i], q=q)

            ax = axs[i] if nplots > 1 else axs
            ax.hist(
                self.chain[i],
                bins=50,
                edgecolor="grey",
                facecolor="linen",
                # lw=2,
                # ls="--",
                # alpha=0.5,
                histtype="stepfilled",
            )

            ax.axvline(quantiles[0], ls="--", c="red", lw=0.5)
            ax.axvline(quantiles[2], ls="--", c="red", lw=0.5)
            ax.axvline(quantiles[1], ls="-", c="red", lw=0.5, label="Best")

            if self.labels is not None:
                ax.set_xlabel(self.labels[i])

            if self.fiducial is not None:
                ax.axvline(self.fiducial[i], ls="-", c="k", lw=0.5, label="Fiducial")

            ax.legend()

        fig.supylabel("MCMC samples")
        return fig, axs

    def report(self):
        q = [0.16, 0.84]

        print()
        for i in range(self.chain.shape[0]):
            quantiles = np.quantile(self.chain[i], q)
            plus = quantiles[1] - self.best[i]
            minus = self.best[i] - quantiles[0]

            if self.labels:
                print(f"{quantiles[1]:.4f} +{plus:.4f}/-{minus:.4f} {self.labels[i]}")
            else:
                print(f"{quantiles[1]:.4f} +{plus:.4f}/-{minus:.4f}")
