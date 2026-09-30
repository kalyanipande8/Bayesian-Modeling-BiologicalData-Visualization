"""
Matplotlib/seaborn visualizations translating the hierarchical model's
posterior distributions into plots a bench scientist can read directly:

1. Posterior distribution plots (EC50, Hill slope) for a representative
   subset of compounds, with the simulated "true" value marked.
2. Dose-response curves with posterior credible-interval bands, overlaid
   on the raw replicate data, for the same subset.
3. A forest plot of posterior EC50 (with 95% credible intervals) across
   all 18 compounds, sorted by potency, colored by whether the interval
   captures the true simulated value.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from simulate_dose_response_data import hill_equation

sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("viridis", 18)


def plot_posterior_distributions(posterior, truth, compound_names, out_path, n_show=6):
    idx_show = np.linspace(0, len(compound_names) - 1, n_show).astype(int)
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 7), squeeze=False)

    ec50_draws = 10 ** posterior["log_ec50"]  # (n_draws, C)
    hill_draws = posterior["hill"]

    for j, c in enumerate(idx_show):
        ax = axes[0, j]
        sns.histplot(ec50_draws[:, c], bins=40, kde=True, ax=ax, color=PALETTE[c])
        ax.axvline(truth.loc[c, "true_ec50"], color="crimson", ls="--", lw=2, label="true EC50")
        ax.set_title(compound_names[c])
        ax.set_xlabel("EC50 (uM)")
        if j == 0:
            ax.set_ylabel("Posterior density")
            ax.legend(fontsize=9)
        else:
            ax.set_ylabel("")

        ax2 = axes[1, j]
        sns.histplot(hill_draws[:, c], bins=40, kde=True, ax=ax2, color=PALETTE[c])
        ax2.axvline(truth.loc[c, "true_hill"], color="crimson", ls="--", lw=2)
        ax2.set_xlabel("Hill slope")
        if j == 0:
            ax2.set_ylabel("Posterior density")
        else:
            ax2.set_ylabel("")

    fig.suptitle("Posterior distributions of EC50 and Hill slope (dashed = simulated truth)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dose_response_curves(posterior, data, truth, compound_names, out_path, n_show=6):
    idx_show = np.linspace(0, len(compound_names) - 1, n_show).astype(int)
    fig, axes = plt.subplots(2, (n_show + 1) // 2, figsize=(6 * ((n_show + 1) // 2), 9), squeeze=False)
    axes = axes.flatten()

    dose_grid = np.logspace(-3, 2, 200)
    n_draws = posterior["log_ec50"].shape[0]
    sample_idx = np.random.default_rng(0).choice(n_draws, size=min(500, n_draws), replace=False)

    for j, c in enumerate(idx_show):
        ax = axes[j]
        curves = np.array(
            [
                hill_equation(
                    dose_grid,
                    posterior["bottom"][i, c],
                    posterior["top"][i, c],
                    10 ** posterior["log_ec50"][i, c],
                    posterior["hill"][i, c],
                )
                for i in sample_idx
            ]
        )
        lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
        med = np.percentile(curves, 50, axis=0)

        ax.fill_between(dose_grid, lo, hi, color=PALETTE[c], alpha=0.3, label="95% credible band")
        ax.plot(dose_grid, med, color=PALETTE[c], lw=2, label="posterior median curve")

        sub = data[data["compound_idx"] == c]
        ax.scatter(sub["dose_uM"], sub["response"], color="black", s=18, alpha=0.6, label="replicates")

        true_curve = hill_equation(
            dose_grid,
            truth.loc[c, "true_bottom"],
            truth.loc[c, "true_top"],
            truth.loc[c, "true_ec50"],
            truth.loc[c, "true_hill"],
        )
        ax.plot(dose_grid, true_curve, color="crimson", lw=1.5, ls="--", label="true curve")

        ax.set_xscale("log")
        ax.set_title(compound_names[c])
        ax.set_xlabel("Dose (uM)")
        ax.set_ylabel("Response")
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")

    for j in range(len(idx_show), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Dose-response curves with 95% posterior credible-interval bands")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ec50_forest(posterior, truth, compound_names, out_path):
    ec50_draws = 10 ** posterior["log_ec50"]
    med = np.percentile(ec50_draws, 50, axis=0)
    lo = np.percentile(ec50_draws, 2.5, axis=0)
    hi = np.percentile(ec50_draws, 97.5, axis=0)
    true_vals = truth["true_ec50"].to_numpy()

    order = np.argsort(med)
    covers = (true_vals[order] >= lo[order]) & (true_vals[order] <= hi[order])

    fig, ax = plt.subplots(figsize=(9, 10))
    y = np.arange(len(order))
    colors = np.where(covers, "#2c7fb8", "#d95f0e")

    ax.hlines(y, lo[order], hi[order], color=colors, lw=2.5)
    ax.plot(med[order], y, "o", color="black", ms=5, label="posterior median")
    ax.plot(true_vals[order], y, "x", color="crimson", ms=8, mew=2, label="true simulated EC50")

    ax.set_yticks(y)
    ax.set_yticklabels([compound_names[i] for i in order])
    ax.set_xscale("log")
    ax.set_xlabel("EC50 (uM, log scale)")
    ax.set_title("Forest plot: posterior EC50 (median + 95% credible interval) across 18 compounds")
    ax.legend(loc="lower right")
    n_covered = int(covers.sum())
    ax.text(
        0.01,
        -0.09,
        f"{n_covered}/{len(order)} compounds' 95% credible intervals contain the true simulated EC50",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trace_diagnostics(posterior, out_path, params=("mu_log_ec50", "mu_hill", "sigma_obs")):
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 3 * len(params)), squeeze=False)
    for i, p in enumerate(params):
        ax = axes[i, 0]
        ax.plot(posterior[p], lw=0.7, color=PALETTE[i % len(PALETTE)])
        ax.set_ylabel(p)
        ax.set_xlabel("MCMC draw (post burn-in, thinned)")
    fig.suptitle("Trace plots of key population-level hyperparameters")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
