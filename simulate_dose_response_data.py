"""
Simulate dose-response (4-parameter logistic / Hill equation) data for 18
compounds screened in a hypothetical potency assay.

Hierarchical structure
-----------------------
Each compound's Hill-equation parameters are drawn from a *shared population
distribution*, rather than being independent:

    log10(EC50_c)  ~ Normal(mu_logEC50, sigma_logEC50)
    hill_c         ~ Normal(mu_hill,   sigma_hill)      (truncated > 0.1)
    top_c          ~ Normal(mu_top,    sigma_top)
    bottom_c       ~ Normal(mu_bottom, sigma_bottom)

This mirrors a real assay-development setting: 18 analog compounds from the
same chemical series are expected to share a family "shape" (Hill slope,
bottom/top asymptotes) while differing in potency (EC50), and the population
distribution is what a hierarchical Bayesian model recovers and exploits for
partial pooling.

Each compound is tested at 8 log-spaced doses with 3 replicates per dose,
plus a small amount of homoscedastic-in-log-response noise.

Everything here is synthetic. No real compound or experimental data is used.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
N_COMPOUNDS = 18
N_DOSES = 8
N_REPLICATES = 3
DOSE_MIN, DOSE_MAX = 1e-3, 1e2  # uM

# Population (hyper-parameter) truth used to simulate the compounds.
POP_TRUTH = dict(
    mu_log_ec50=0.0,      # log10(EC50) center -> EC50 = 1 uM
    sigma_log_ec50=0.5,   # compound-to-compound spread in log10(EC50)
    mu_hill=1.2,
    sigma_hill=0.25,
    mu_top=100.0,
    sigma_top=5.0,
    mu_bottom=2.0,
    sigma_bottom=2.0,
    sigma_obs=4.0,        # residual (replicate) observation noise, response units
)


def hill_equation(dose, bottom, top, ec50, hill):
    """4-parameter logistic / Hill equation."""
    return bottom + (top - bottom) / (1.0 + (ec50 / dose) ** hill)


def simulate(seed: int = RNG_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    data : long-format DataFrame with one row per (compound, dose, replicate)
    truth : DataFrame with the true simulated per-compound parameters
            (used later only for the sanity-check test, never for fitting).
    """
    rng = np.random.default_rng(seed)

    doses = np.logspace(np.log10(DOSE_MIN), np.log10(DOSE_MAX), N_DOSES)

    compound_ids = [f"CMPD-{i+1:02d}" for i in range(N_COMPOUNDS)]

    log_ec50 = rng.normal(POP_TRUTH["mu_log_ec50"], POP_TRUTH["sigma_log_ec50"], N_COMPOUNDS)
    ec50 = 10 ** log_ec50
    hill = np.clip(
        rng.normal(POP_TRUTH["mu_hill"], POP_TRUTH["sigma_hill"], N_COMPOUNDS), 0.2, None
    )
    top = rng.normal(POP_TRUTH["mu_top"], POP_TRUTH["sigma_top"], N_COMPOUNDS)
    bottom = rng.normal(POP_TRUTH["mu_bottom"], POP_TRUTH["sigma_bottom"], N_COMPOUNDS)

    truth = pd.DataFrame(
        {
            "compound": compound_ids,
            "true_ec50": ec50,
            "true_log_ec50": log_ec50,
            "true_hill": hill,
            "true_top": top,
            "true_bottom": bottom,
        }
    )

    rows = []
    for c_idx, cid in enumerate(compound_ids):
        mean_response = hill_equation(doses, bottom[c_idx], top[c_idx], ec50[c_idx], hill[c_idx])
        for d_idx, dose in enumerate(doses):
            noisy = mean_response[d_idx] + rng.normal(0, POP_TRUTH["sigma_obs"], N_REPLICATES)
            for rep in range(N_REPLICATES):
                rows.append(
                    {
                        "compound": cid,
                        "compound_idx": c_idx,
                        "dose_uM": dose,
                        "replicate": rep + 1,
                        "response": noisy[rep],
                    }
                )

    data = pd.DataFrame(rows)
    return data, truth


if __name__ == "__main__":
    data, truth = simulate()
    data.to_csv("output/simulated_dose_response.csv", index=False)
    truth.to_csv("output/simulated_truth.csv", index=False)
    print(f"Simulated {data['compound'].nunique()} compounds, {len(data)} total measurements.")
    print(truth.head())
