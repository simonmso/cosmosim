import numpy as np
import matplotlib.pyplot as plt
from Global import APS_COL_W as apsw


class MCMCResult:
    def __init__(self, chain, log_prob, labels=None):
        self.chain = np.array(chain)
        self.log_prob = log_prob
        self.labels = labels

    def hist2d(self):
        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

        ax.hist2d(self.chain[0], self.chain[1], norm="log", bins=100)

        if self.labels is not None:
            ax.set_xlabel(self.labels[0])
            ax.set_ylabel(self.labels[1])

        ax.set_title("MCMC Results")

        return fig, ax

    def contour(self):
        fig, ax = plt.subplots(figsize=(apsw, 0.7 * apsw))

        q = [1.0 - 0.68, 1.0 - 0.95]
        one_sig, two_sig = np.quantile(self.log_prob, q)

        one_sig_idx = self.log_prob > one_sig
        two_sig_idx = self.log_prob > two_sig

        ax.scatter(self.chain[0][two_sig_idx], self.chain[1][two_sig_idx])

        ax.scatter(self.chain[0][one_sig_idx], self.chain[1][one_sig_idx])

        if self.labels is not None:
            ax.set_xlabel(self.labels[0])
            ax.set_ylabel(self.labels[1])

        ax.set_title("MCMC Contours")
        return fig, ax

    def hist(self):
        nplots = self.chain.shape[0]
        fig, axs = plt.subplots(nrows=nplots, figsize=(apsw, 0.5 * nplots * apsw))

        for i in range(nplots):
            ax = axs[i]
            ax.hist(
                self.chain[i],
                bins=50,
            )
            ax.set_xlabel(self.labels[i])

        return fig, axs
