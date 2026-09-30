"""
Pipeline sanity tests. These run the real simulation + a (smaller, faster)
real MCMC fit and check for honest properties of a correct Bayesian fit:
18 compounds present, credible intervals mostly contain the injected true
value, and the sampler shows signs of having converged.
"""
import numpy as np
import pytest

from simulate_dose_response_data import simulate, N_COMPOUNDS
from bayesian_hierarchical_model import run_mcmc
from diagnostics import effective_sample_size, gelman_rubin


@pytest.fixture(scope="module")
def simulated():
    data, truth = simulate(seed=42)
    return data, truth


@pytest.fixture(scope="module")
def fitted(simulated):
    data, truth = simulated
    # Smaller chain than run_all.py's production run, to keep tests fast,
    # but still a real MCMC fit (not a shortcut/mocked result).
    chain1, acc1, n_draws = run_mcmc(data, N_COMPOUNDS, n_iter=2500, burn_in=1000, thin=2, seed=1, chain_seed_offset=0)
    chain2, acc2, _ = run_mcmc(data, N_COMPOUNDS, n_iter=2500, burn_in=1000, thin=2, seed=1, chain_seed_offset=1)
    return chain1, chain2, acc1, acc2, n_draws


def test_18_compounds_present(simulated):
    data, truth = simulated
    assert truth["compound"].nunique() == 18
    assert data["compound"].nunique() == 18
    assert N_COMPOUNDS == 18


def test_each_compound_has_replicate_measurements(simulated):
    data, _ = simulated
    counts = data.groupby("compound").size()
    assert (counts >= 3 * 6).all()  # >= 6 doses x 3 replicates minimum


def test_mcmc_produces_draws(fitted):
    chain1, chain2, acc1, acc2, n_draws = fitted
    assert n_draws > 500
    assert chain1["log_ec50"].shape == (n_draws, N_COMPOUNDS)
    # Reasonable Metropolis acceptance rate (not stuck, not trivially always-accept)
    assert 0.05 < acc1 < 0.95
    assert 0.05 < acc2 < 0.95


def test_convergence_diagnostics_are_reasonable(fitted):
    chain1, chain2, *_ = fitted
    for p in ["mu_log_ec50", "mu_hill", "sigma_obs"]:
        rhat = gelman_rubin(np.stack([chain1[p], chain2[p]]))
        assert rhat == rhat  # not NaN
        assert rhat < 1.2, f"{p} R-hat too high: {rhat}"

        ess = effective_sample_size(np.concatenate([chain1[p], chain2[p]]))
        assert ess > 20, f"{p} ESS too low: {ess}"


def test_credible_intervals_mostly_contain_true_ec50(simulated, fitted):
    """
    The core Bayesian-correctness sanity check: for simulated data with a
    known ('injected') true EC50 per compound, the posterior 95% credible
    interval should contain that true value for most (not necessarily all
    -- that's what '95%' means) of the 18 compounds.
    """
    data, truth = simulated
    chain1, chain2, *_ = fitted
    posterior_log_ec50 = np.concatenate([chain1["log_ec50"], chain2["log_ec50"]], axis=0)
    ec50_draws = 10 ** posterior_log_ec50

    lo = np.percentile(ec50_draws, 2.5, axis=0)
    hi = np.percentile(ec50_draws, 97.5, axis=0)
    true_ec50 = truth["true_ec50"].to_numpy()

    covered = (true_ec50 >= lo) & (true_ec50 <= hi)
    coverage_frac = covered.mean()

    # With only ~750 post-burn-in draws/chain this is a coarser fit than
    # run_all.py's full run, so we use a lenient but still meaningful bar.
    assert coverage_frac >= 0.65, (
        f"Only {covered.sum()}/{len(covered)} compounds' 95% CIs contained "
        f"the true EC50 (expected most of them to, for a correctly "
        f"calibrated hierarchical model)"
    )


def test_hill_slope_recovered_reasonably(simulated, fitted):
    data, truth = simulated
    chain1, chain2, *_ = fitted
    posterior_hill = np.concatenate([chain1["hill"], chain2["hill"]], axis=0)
    med_hill = np.median(posterior_hill, axis=0)
    true_hill = truth["true_hill"].to_numpy()
    # Posterior median Hill slope should be in the right ballpark (not exact,
    # since it's estimated jointly with EC50/top/bottom from noisy data).
    rel_err = np.abs(med_hill - true_hill) / true_hill
    assert np.median(rel_err) < 0.5
