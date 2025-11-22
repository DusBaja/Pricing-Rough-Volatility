# products.py
import numpy as np
from observation import ObservationFrequency, get_observation_indices

def price_autocall_athena_from_paths(
    S_paths: np.ndarray,
    df_paths: np.ndarray,
    nominal: float,
    strike: float,
    coupon_per_year: float,
    maturity_years: float,
    obs_freq: str,
    call_barrier: float,
    protection_barrier: float,
    steps_per_year: int,
):
    """
    Generic Athena autocall payoff using pre-simulated S_paths, df_paths.

    S_paths: (n_paths, n_steps+1)
    df_paths: (n_paths, n_steps+1)
    """
    freq_map = {
        "annual": ObservationFrequency.ANNUAL,
        "monthly": ObservationFrequency.MONTHLY,
        "daily": ObservationFrequency.DAILY,
    }
    freq = freq_map[obs_freq.lower()]
    obs_indices, n_steps = get_observation_indices(maturity_years, steps_per_year, freq)
    dt = maturity_years / n_steps

    n_paths = S_paths.shape[0]
    call_level = call_barrier * strike
    prot_level = protection_barrier * strike

    payoffs = np.zeros(n_paths)
    discounts = np.zeros(n_paths)

    for i in range(n_paths):
        path_S = S_paths[i]
        path_df = df_paths[i]

        autocalled = False
        for idx in obs_indices[:-1]:
            if path_S[idx] >= call_level:
                t_years = idx * dt
                payoff = nominal * (1.0 + coupon_per_year * t_years)
                payoffs[i] = payoff
                discounts[i] = path_df[idx]
                autocalled = True
                break

        if not autocalled:
            idx_T = obs_indices[-1]
            ST = path_S[idx_T]
            DF_T = path_df[idx_T]
            t_years = idx_T * dt

            if ST >= prot_level:
                payoff = nominal * (1.0 + coupon_per_year * t_years)
            else:
                payoff = nominal * (ST / strike)

            payoffs[i] = payoff
            discounts[i] = DF_T

    discounted = payoffs * discounts
    price = np.mean(discounted)
    std_error = np.std(discounted, ddof=1) / np.sqrt(n_paths)
    return price, std_error
