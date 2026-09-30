"""
Hand-written Bayesian hierarchical model for dose-response (Hill equation)
data, fit jointly across all compounds via a Metropolis-within-Gibbs MCMC
sampler implemented from scratch in numpy/scipy.

Why hand-written instead of PyMC/Stan
--------------------------------------
See README.md for the full explanation. In short: the machine this was
developed on had essentially no free disk space (a few hundred MB) at the
time of development, which is not enough headroom to safely install PyMC's
dependency stack (PyTensor, a C/BLAS toolchain, compiled binaries) without
risking the disk filling up entirely. Rather than skip the modeling work,
this implements the *same* hierarchical model and inference target
(posterior over per-compound Hill-equation parameters, partially pooled
through population-level hyperparameters) using a Metropolis-within-Gibbs
sampler built directly on numpy/scipy. This is real MCMC — not a shortcut
like a bootstrap or a plug-in point estimate — and every conjugate update
below is derived by hand from the normal-normal / normal-inverse-gamma
identities.

Model
-----
For compound c = 1..C, dose d (uM), replicate r:

    response[c, d, r] ~ Normal(mu[c, d], sigma_obs^2)
    mu[c, d] = bottom_c + (top_c - bottom_c) / (1 + (ec50_c / dose_d) ** hill_c)

Hierarchical (population) priors, i.e. partial pooling across compounds:

    log10(ec50_c) ~ Normal(mu_log_ec50, sigma_log_ec50^2)
    hill_c        ~ Normal(mu_hill,     sigma_hill^2)          [rejected if <= 0]
    top_c         ~ Normal(mu_top,      sigma_top^2)
    bottom_c      ~ Normal(mu_bottom,   sigma_bottom^2)

Hyperpriors (weakly informative, conjugate families):

    mu_log_ec50 ~ Normal(0, 10^2)         sigma_log_ec50^2 ~ InvGamma(2, 1)
    mu_hill     ~ Normal(1, 10^2)         sigma_hill^2     ~ InvGamma(2, 1)
    mu_top      ~ Normal(100, 100^2)      sigma_top^2      ~ InvGamma(2, 50)
    mu_bottom   ~ Normal(0, 100^2)        sigma_bottom^2   ~ InvGamma(2, 5)
    sigma_obs^2 ~ InvGamma(2, 10)

Sampling scheme (Metropolis-within-Gibbs)
------------------------------------------
1. For each compound c, update (log_ec50_c, hill_c, top_c, bottom_c) jointly
   with a random-walk Metropolis step, using the Hill-equation likelihood
   for that compound's data plus the current population Normal priors.
2. Given all compound-level parameters, update each population mean
   (mu_log_ec50, mu_hill, mu_top, mu_bottom) by an exact conjugate
   Normal-Normal draw.
3. Given all compound-level parameters and the means, update each
   population variance by an exact conjugate Normal-InverseGamma draw.
4. Update sigma_obs^2 by an exact conjugate InverseGamma draw from the
   pooled residual sum of squares across all compounds/doses/replicates.

Steps 2-4 are exact Gibbs draws (no tuning needed); step 1 is a tuned
random-walk Metropolis step per compound.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from simulate_dose_response_data import hill_equation


def _invgamma_rvs(rng, shape, scale):
    """Sample from InverseGamma(shape, scale) via scipy's parameterization."""
    return stats.invgamma.rvs(a=shape, scale=scale, random_state=rng)


class HierarchicalHillModel:
    """Metropolis-within-Gibbs sampler for the hierarchical Hill-equation model."""

    def __init__(self, data, n_compounds: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_compounds = n_compounds

        # Pre-index the data per compound for fast per-compound likelihood evals.
        self.doses = []
        self.responses = []
        for c in range(n_compounds):
            sub = data[data["compound_idx"] == c]
            self.doses.append(sub["dose_uM"].to_numpy())
            self.responses.append(sub["response"].to_numpy())
        self.n_obs_total = sum(len(r) for r in self.responses)

        # Hyperprior parameters (fixed).
        self.hp = dict(
            mu_log_ec50_prior=(0.0, 10.0),
            mu_hill_prior=(1.0, 10.0),
            mu_top_prior=(100.0, 100.0),
            mu_bottom_prior=(0.0, 100.0),
            sigma_log_ec50_ig=(2.0, 1.0),
            sigma_hill_ig=(2.0, 1.0),
            sigma_top_ig=(2.0, 50.0),
            sigma_bottom_ig=(2.0, 5.0),
            sigma_obs_ig=(2.0, 10.0),
        )

        # Initialize compound-level parameters from crude data-driven guesses.
        self.log_ec50 = np.zeros(n_compounds)
        self.hill = np.full(n_compounds, 1.0)
        self.top = np.zeros(n_compounds)
        self.bottom = np.zeros(n_compounds)
        for c in range(n_compounds):
            resp = self.responses[c]
            self.top[c] = np.percentile(resp, 90)
            self.bottom[c] = np.percentile(resp, 10)
            self.log_ec50[c] = 0.0
            self.hill[c] = 1.0

        # Initialize hyperparameters.
        self.mu_log_ec50 = 0.0
        self.mu_hill = 1.0
        self.mu_top = float(np.mean(self.top))
        self.mu_bottom = float(np.mean(self.bottom))
        self.sigma_log_ec50 = 0.5
        self.sigma_hill = 0.3
        self.sigma_top = 5.0
        self.sigma_bottom = 3.0
        self.sigma_obs = 5.0

        # Random-walk proposal scales per compound-level parameter (tuned by hand).
        self.prop_scale = np.array([0.08, 0.10, 2.0, 1.5])  # log_ec50, hill, top, bottom

    # ---------------------------------------------------------------
    def _log_lik_compound(self, c, log_ec50, hill, top, bottom):
        if hill <= 0:
            return -np.inf
        ec50 = 10 ** log_ec50
        pred = hill_equation(self.doses[c], bottom, top, ec50, hill)
        resid = self.responses[c] - pred
        return float(np.sum(stats.norm.logpdf(resid, loc=0.0, scale=self.sigma_obs)))

    def _log_prior_compound(self, log_ec50, hill, top, bottom):
        if hill <= 0:
            return -np.inf
        lp = 0.0
        lp += stats.norm.logpdf(log_ec50, self.mu_log_ec50, self.sigma_log_ec50)
        lp += stats.norm.logpdf(hill, self.mu_hill, self.sigma_hill)
        lp += stats.norm.logpdf(top, self.mu_top, self.sigma_top)
        lp += stats.norm.logpdf(bottom, self.mu_bottom, self.sigma_bottom)
        return lp

    def _mh_update_compound(self, c):
        cur = np.array([self.log_ec50[c], self.hill[c], self.top[c], self.bottom[c]])
        cur_logp = self._log_lik_compound(c, *cur) + self._log_prior_compound(*cur)

        prop = cur + self.rng.normal(0, self.prop_scale)
        prop_logp = self._log_lik_compound(c, *prop) + self._log_prior_compound(*prop)

        accept_ratio = prop_logp - cur_logp
        accepted = np.log(self.rng.uniform()) < accept_ratio
        if accepted:
            self.log_ec50[c], self.hill[c], self.top[c], self.bottom[c] = prop
        return accepted

    # ---------------------------------------------------------------
    def _gibbs_update_pop_mean(self, values, sigma_pop, prior_mean, prior_sd):
        """Exact Normal-Normal conjugate update for a population mean."""
        n = len(values)
        prior_prec = 1.0 / prior_sd**2
        data_prec = n / sigma_pop**2
        post_var = 1.0 / (prior_prec + data_prec)
        post_mean = post_var * (prior_mean * prior_prec + np.sum(values) * (1.0 / sigma_pop**2))
        return self.rng.normal(post_mean, np.sqrt(post_var))

    def _gibbs_update_pop_sigma(self, values, pop_mean, ig_shape, ig_scale):
        """Exact Normal-InverseGamma conjugate update for a population variance."""
        n = len(values)
        resid = values - pop_mean
        post_shape = ig_shape + n / 2.0
        post_scale = ig_scale + 0.5 * np.sum(resid**2)
        var = _invgamma_rvs(self.rng, post_shape, post_scale)
        return np.sqrt(var)

    def _gibbs_update_sigma_obs(self):
        ss = 0.0
        for c in range(self.n_compounds):
            ec50 = 10 ** self.log_ec50[c]
            pred = hill_equation(self.doses[c], self.bottom[c], self.top[c], ec50, self.hill[c])
            ss += float(np.sum((self.responses[c] - pred) ** 2))
        shape0, scale0 = self.hp["sigma_obs_ig"]
        post_shape = shape0 + self.n_obs_total / 2.0
        post_scale = scale0 + 0.5 * ss
        var = _invgamma_rvs(self.rng, post_shape, post_scale)
        return np.sqrt(var)

    # ---------------------------------------------------------------
    def step(self):
        accepts = 0
        for c in range(self.n_compounds):
            accepts += self._mh_update_compound(c)

        m0, s0 = self.hp["mu_log_ec50_prior"]
        self.mu_log_ec50 = self._gibbs_update_pop_mean(self.log_ec50, self.sigma_log_ec50, m0, s0)
        m0, s0 = self.hp["mu_hill_prior"]
        self.mu_hill = self._gibbs_update_pop_mean(self.hill, self.sigma_hill, m0, s0)
        m0, s0 = self.hp["mu_top_prior"]
        self.mu_top = self._gibbs_update_pop_mean(self.top, self.sigma_top, m0, s0)
        m0, s0 = self.hp["mu_bottom_prior"]
        self.mu_bottom = self._gibbs_update_pop_mean(self.bottom, self.sigma_bottom, m0, s0)

        a0, b0 = self.hp["sigma_log_ec50_ig"]
        self.sigma_log_ec50 = self._gibbs_update_pop_sigma(self.log_ec50, self.mu_log_ec50, a0, b0)
        a0, b0 = self.hp["sigma_hill_ig"]
        self.sigma_hill = self._gibbs_update_pop_sigma(self.hill, self.mu_hill, a0, b0)
        a0, b0 = self.hp["sigma_top_ig"]
        self.sigma_top = self._gibbs_update_pop_sigma(self.top, self.mu_top, a0, b0)
        a0, b0 = self.hp["sigma_bottom_ig"]
        self.sigma_bottom = self._gibbs_update_pop_sigma(self.bottom, self.mu_bottom, a0, b0)

        self.sigma_obs = self._gibbs_update_sigma_obs()

        return accepts / self.n_compounds

    # ---------------------------------------------------------------
    def snapshot(self):
        return dict(
            log_ec50=self.log_ec50.copy(),
            hill=self.hill.copy(),
            top=self.top.copy(),
            bottom=self.bottom.copy(),
            mu_log_ec50=self.mu_log_ec50,
            mu_hill=self.mu_hill,
            mu_top=self.mu_top,
            mu_bottom=self.mu_bottom,
            sigma_log_ec50=self.sigma_log_ec50,
            sigma_hill=self.sigma_hill,
            sigma_top=self.sigma_top,
            sigma_bottom=self.sigma_bottom,
            sigma_obs=self.sigma_obs,
        )


def run_mcmc(data, n_compounds, n_iter=6000, burn_in=2000, thin=2, seed=0, chain_seed_offset=0):
    model = HierarchicalHillModel(data, n_compounds, seed=seed + chain_seed_offset)
    draws = []
    accept_rates = []
    for it in range(n_iter):
        rate = model.step()
        if it >= burn_in and (it - burn_in) % thin == 0:
            draws.append(model.snapshot())
        accept_rates.append(rate)

    # Reassemble into arrays keyed by parameter name.
    n_draws = len(draws)
    out = {
        "log_ec50": np.stack([d["log_ec50"] for d in draws]),   # (n_draws, C)
        "hill": np.stack([d["hill"] for d in draws]),
        "top": np.stack([d["top"] for d in draws]),
        "bottom": np.stack([d["bottom"] for d in draws]),
        "mu_log_ec50": np.array([d["mu_log_ec50"] for d in draws]),
        "mu_hill": np.array([d["mu_hill"] for d in draws]),
        "mu_top": np.array([d["mu_top"] for d in draws]),
        "mu_bottom": np.array([d["mu_bottom"] for d in draws]),
        "sigma_log_ec50": np.array([d["sigma_log_ec50"] for d in draws]),
        "sigma_hill": np.array([d["sigma_hill"] for d in draws]),
        "sigma_top": np.array([d["sigma_top"] for d in draws]),
        "sigma_bottom": np.array([d["sigma_bottom"] for d in draws]),
        "sigma_obs": np.array([d["sigma_obs"] for d in draws]),
    }
    mean_accept = float(np.mean(accept_rates))
    return out, mean_accept, n_draws
