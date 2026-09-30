# Bayesian Statistical Modeling & Visualization for Biological Data

Bayesian hierarchical modeling of dose-response (Hill-equation) curves for
18 compounds, with real posterior distributions, credible intervals, MCMC
convergence diagnostics, and matplotlib/seaborn visualizations built to
translate posteriors for experimental scientists.

> **Disclaimer:** This is a portfolio/demo project. All dose-response data
> is **synthetically simulated** by `simulate_dose_response_data.py` with a
> known, injected hierarchical structure. No real compound or experimental
> data is used.

## Honest note: PyMC vs. hand-written MCMC

This project was built on a machine with essentially no free disk space
(`df -h /` showed as little as ~190Mi-350Mi free at points during
development, out of a 228Gi volume that was otherwise full). PyMC's
dependency stack (PyTensor plus a C/BLAS compiler toolchain and compiled
binaries) is not something that can be safely installed with that little
headroom — a failed or partial install risks filling the disk entirely.

Rather than fake the Bayesian modeling or fall back to a non-Bayesian
shortcut (e.g., per-compound least-squares fits with bootstrap CIs), this
project implements a **real Bayesian hierarchical model fit via a
hand-written Metropolis-within-Gibbs MCMC sampler**, built directly on
`numpy`/`scipy`, in `bayesian_hierarchical_model.py`:

- Compound-level Hill-equation parameters (EC50, Hill slope, top, bottom)
  are updated with a tuned random-walk **Metropolis** step, using the
  actual nonlinear Hill-equation likelihood.
- Population-level hyperparameters (the pooled mean/variance each
  compound's parameters are drawn from) are updated with **exact conjugate
  Gibbs draws** (Normal-Normal for means, Normal-Inverse-Gamma for
  variances), derived by hand from the standard conjugacy identities.
- Convergence diagnostics (Gelman-Rubin R-hat, effective sample size via
  Geyer's initial-monotone-sequence estimator) are also implemented from
  scratch in `diagnostics.py`, since ArviZ was not installed either.

This is genuine MCMC inference over a genuine hierarchical model — the same
statistical target PyMC would fit for this problem — not a point-estimate
or bootstrap shortcut. If disk space were not constrained, the natural
next step would be to port `bayesian_hierarchical_model.py` to
`pm.Model()` + NUTS, which would sample faster and scale better to bigger
hierarchies; the model specification (priors, likelihood, pooling
structure) would be unchanged.

## The model

For compound *c*, dose *d* (µM), replicate *r*:

```
response[c, d, r] ~ Normal(mu[c, d], sigma_obs^2)
mu[c, d] = bottom_c + (top_c - bottom_c) / (1 + (EC50_c / dose_d) ** hill_c)
```

Hierarchical (partially-pooled) priors across all 18 compounds:

```
log10(EC50_c) ~ Normal(mu_log_ec50, sigma_log_ec50^2)
hill_c        ~ Normal(mu_hill,     sigma_hill^2)
top_c         ~ Normal(mu_top,      sigma_top^2)
bottom_c      ~ Normal(mu_bottom,   sigma_bottom^2)
```

with weakly-informative Normal/Inverse-Gamma hyperpriors on the population
means and variances. This is the standard structure for "many related
dose-response curves from the same assay/chemical series" — each
compound's curve is fit using its own data *and* what all 17 other
compounds imply about plausible Hill slopes and asymptotes (partial
pooling), which is exactly what makes a hierarchical model more useful
than fitting each compound in isolation.

## Pipeline

1. **`simulate_dose_response_data.py`** — simulates 18 compounds x 8
   log-spaced doses x 3 replicates, with compound-level Hill parameters
   drawn from a shared population distribution (injected ground truth
   saved for the sanity-check test).
2. **`bayesian_hierarchical_model.py`** — the hand-written
   Metropolis-within-Gibbs sampler described above; runs 2 independent
   chains for R-hat.
3. **`diagnostics.py`** — R-hat, effective sample size, trace stability.
4. **`visualize_posteriors.py`** — matplotlib/seaborn plots:
   posterior distributions (EC50, Hill slope) with the true value marked,
   dose-response curves with 95% posterior credible-interval bands over
   the raw replicate data, a forest plot of EC50 across all 18 compounds,
   and trace plots of the population hyperparameters.
5. **`run_all.py`** — runs the whole pipeline end-to-end and writes
   `output/`.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
pytest -q
```

## Results

_Numbers below are copied directly from an actual run of `python
run_all.py` (fixed random seeds, so reproducible)._ See
`output/run_summary.json`, `output/posterior_summary.csv`, and
`output/convergence_diagnostics.csv` for the full machine-readable output.

- **18 compounds, 432 measurements** (8 doses x 3 replicates x 18
  compounds) simulated and fit jointly.
- **2 independent MCMC chains**, 6000 iterations each (2000 burn-in, thin
  2) -> 2000 post-burn-in draws/chain each (4000 pooled), in 115.8s total
  on a laptop CPU. Mean compound-level Metropolis acceptance rate: 0.17 and
  0.16 for the two chains (healthy for a 4-parameter random-walk proposal;
  textbook target range is roughly 0.15-0.4).
- **Convergence:** max Gelman-Rubin **R-hat = 1.006** across all 7 reported
  population-level hyperparameters (< 1.1 is the conventional
  "converged" threshold — comfortably passed). Min **effective sample
  size = 447** draws (out of 4000 pooled draws) — `mu_bottom` is the
  slowest-mixing parameter (ESS 447, half-stability 0.36) since the data
  only weakly identify the population spread of the bottom asymptote;
  every other hyperparameter has ESS > 900. Reported honestly, not rounded
  up — see `output/convergence_diagnostics.csv` for the full per-parameter
  table.
- **Credible-interval coverage: 18/18 (100%)** of the 18 compounds' 95%
  posterior credible intervals for EC50 contained the true simulated
  EC50 (`output/posterior_summary.csv`). With only 18 compounds, seeing
  100% instead of the nominal ~95% is well within sampling noise (one
  miss out of 18 would already be ~94%); it is not evidence of
  miscalibration, just a small-N draw. Hill-slope posterior medians also
  tracked the true simulated values closely for all 18 compounds (e.g.
  CMPD-05: true 1.506 vs. posterior median 1.496; CMPD-13: true 1.735 vs.
  1.607).
- Visualizations saved to `output/`: `posterior_distributions.png`,
  `dose_response_curves.png`, `ec50_forest_plot.png`,
  `trace_diagnostics.png`.

Run `python run_all.py` yourself to reproduce the exact console output and
regenerate all files in `output/`.
