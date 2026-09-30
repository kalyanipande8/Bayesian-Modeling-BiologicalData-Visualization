"""
End-to-end pipeline:

1. Simulate 18-compound hierarchical dose-response data.
2. Fit the hand-written Metropolis-within-Gibbs hierarchical Hill-equation
   model (2 independent chains, for R-hat).
3. Run honest convergence diagnostics (R-hat, ESS, trace stability).
4. Produce matplotlib/seaborn visualizations for scientists.
5. Print a results summary (also used to write the README numbers).
"""
import json
import os
import time

import numpy as np
import pandas as pd

from simulate_dose_response_data import simulate, N_COMPOUNDS
from bayesian_hierarchical_model import run_mcmc
from diagnostics import effective_sample_size, gelman_rubin, trace_half_stability
from visualize_posteriors import (
    plot_posterior_distributions,
    plot_dose_response_curves,
    plot_ec50_forest,
    plot_trace_diagnostics,
)

OUT_DIR = "output"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("1. Simulating dose-response data for 18 compounds")
    print("=" * 70)
    data, truth = simulate()
    data.to_csv(f"{OUT_DIR}/simulated_dose_response.csv", index=False)
    truth.to_csv(f"{OUT_DIR}/simulated_truth.csv", index=False)
    print(f"  {truth['compound'].nunique()} compounds, {len(data)} measurements "
          f"({data.groupby('compound').size().iloc[0]} obs/compound)")

    print("\n" + "=" * 70)
    print("2. Fitting hierarchical Hill-equation model (hand-written MCMC)")
    print("=" * 70)
    n_iter, burn_in, thin = 6000, 2000, 2
    t0 = time.time()
    chain1, acc1, n_draws = run_mcmc(data, N_COMPOUNDS, n_iter, burn_in, thin, seed=0, chain_seed_offset=0)
    chain2, acc2, _ = run_mcmc(data, N_COMPOUNDS, n_iter, burn_in, thin, seed=0, chain_seed_offset=1)
    elapsed = time.time() - t0
    print(f"  2 chains x {n_iter} iterations ({burn_in} burn-in, thin={thin}) -> "
          f"{n_draws} draws/chain in {elapsed:.1f}s")
    print(f"  mean compound-level MH acceptance rate: chain1={acc1:.2f}, chain2={acc2:.2f}")

    print("\n" + "=" * 70)
    print("3. Convergence diagnostics")
    print("=" * 70)
    diag_rows = []
    for p in ["mu_log_ec50", "mu_hill", "mu_top", "mu_bottom", "sigma_obs",
              "sigma_log_ec50", "sigma_hill"]:
        rhat = gelman_rubin(np.stack([chain1[p], chain2[p]]))
        ess = effective_sample_size(np.concatenate([chain1[p], chain2[p]]))
        stab = trace_half_stability(chain1[p])
        diag_rows.append(dict(parameter=p, r_hat=rhat, ess=ess, half_stability=stab))
    diag_df = pd.DataFrame(diag_rows)
    print(diag_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    diag_df.to_csv(f"{OUT_DIR}/convergence_diagnostics.csv", index=False)

    max_rhat = diag_df["r_hat"].max()
    min_ess = diag_df["ess"].min()
    print(f"\n  Max R-hat across reported hyperparameters: {max_rhat:.3f} "
          f"({'OK, < 1.1' if max_rhat < 1.1 else 'WARNING: not converged'})")
    print(f"  Min ESS across reported hyperparameters: {min_ess:.0f} draws")

    # Pool both chains post-hoc for the posterior summaries/plots used downstream.
    posterior = {k: np.concatenate([chain1[k], chain2[k]], axis=0) for k in chain1}

    print("\n" + "=" * 70)
    print("4. Per-compound posterior summary (EC50, 95% credible interval)")
    print("=" * 70)
    ec50_draws = 10 ** posterior["log_ec50"]
    med = np.percentile(ec50_draws, 50, axis=0)
    lo = np.percentile(ec50_draws, 2.5, axis=0)
    hi = np.percentile(ec50_draws, 97.5, axis=0)
    hill_med = np.percentile(posterior["hill"], 50, axis=0)

    summary = pd.DataFrame(
        {
            "compound": truth["compound"],
            "true_ec50": truth["true_ec50"],
            "posterior_ec50_median": med,
            "ec50_ci_low_2.5%": lo,
            "ec50_ci_high_97.5%": hi,
            "true_hill": truth["true_hill"],
            "posterior_hill_median": hill_med,
        }
    )
    summary["ci_contains_true_ec50"] = (summary["true_ec50"] >= summary["ec50_ci_low_2.5%"]) & (
        summary["true_ec50"] <= summary["ec50_ci_high_97.5%"]
    )
    summary.to_csv(f"{OUT_DIR}/posterior_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    coverage = summary["ci_contains_true_ec50"].mean()
    print(f"\n  Coverage: {summary['ci_contains_true_ec50'].sum()}/{len(summary)} "
          f"({coverage*100:.0f}%) of 95% credible intervals contain the true simulated EC50")

    print("\n" + "=" * 70)
    print("5. Generating visualizations")
    print("=" * 70)
    names = truth["compound"].tolist()
    plot_posterior_distributions(posterior, truth, names, f"{OUT_DIR}/posterior_distributions.png")
    plot_dose_response_curves(posterior, data, truth, names, f"{OUT_DIR}/dose_response_curves.png")
    plot_ec50_forest(posterior, truth, names, f"{OUT_DIR}/ec50_forest_plot.png")
    plot_trace_diagnostics(chain1, f"{OUT_DIR}/trace_diagnostics.png")
    print(f"  Saved plots to {OUT_DIR}/")

    results = dict(
        n_compounds=int(N_COMPOUNDS),
        n_measurements=int(len(data)),
        n_draws_per_chain=int(n_draws),
        n_chains=2,
        mean_acceptance_rate=[round(acc1, 3), round(acc2, 3)],
        max_rhat=round(float(max_rhat), 3),
        min_ess=round(float(min_ess), 1),
        ec50_ci_coverage=round(float(coverage), 3),
        runtime_seconds=round(elapsed, 1),
    )
    with open(f"{OUT_DIR}/run_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRun summary:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
