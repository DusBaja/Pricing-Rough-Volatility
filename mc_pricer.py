# mc_pricer.py
from vol_models import GBMModel, HestonModel
from rate_models import FlatRateModel, HullWhiteModel
from hybrid_models import HestonHullWhiteHybrid
from products import price_autocall_athena_from_paths

def price_autocall_mc_classic(
    nominal, strike, coupon_per_year,
    maturity_years, obs_freq,
    s0, r0, sigma,
    steps_per_year, n_paths,
    seed=None, antithetic=False,
):
    gbm = GBMModel(s0=s0, r0=r0, sigma=sigma,
                   steps_per_year=steps_per_year, n_paths=n_paths, seed=seed)
    S, df = gbm.simulate_paths(maturity_years, antithetic=antithetic)

    price, se = price_autocall_athena_from_paths(
        S_paths=S,
        df_paths=df,
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_per_year,
        maturity_years=maturity_years,
        obs_freq=obs_freq,
        call_barrier=1.0,
        protection_barrier=0.6,
        steps_per_year=steps_per_year,
    )
    return price, se


def price_autocall_with_vol_model(
    nominal, strike, coupon_per_year,
    maturity_years, obs_freq,
    s0, v0, r0,
    kappa, theta, xi, rho_sv,
    steps_per_year, n_paths,
    seed=None, antithetic=False,
):
    heston = HestonModel(
        s0=s0, v0=v0, r0=r0,
        kappa=kappa, theta=theta, xi=xi, rho_sv=rho_sv,
        steps_per_year=steps_per_year, n_paths=n_paths, seed=seed,
    )
    S, df = heston.simulate_paths(maturity_years, antithetic=antithetic)

    price, se = price_autocall_athena_from_paths(
        S_paths=S,
        df_paths=df,
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_per_year,
        maturity_years=maturity_years,
        obs_freq=obs_freq,
        call_barrier=1.0,
        protection_barrier=0.6,
        steps_per_year=steps_per_year,
    )
    return price, se


def price_autocall_with_rate_model(
    nominal, strike, coupon_per_year,
    maturity_years, obs_freq,
    s0, sigma,
    r0, a, b, sigma_r,
    steps_per_year, n_paths,
    seed=None, antithetic=False,
):
    # rate model: Hull–White
    hw = HullWhiteModel(r0=r0, a=a, b=b, sigma_r=sigma_r)
    r_paths, df = hw.simulate_paths(maturity_years, steps_per_year, n_paths, seed=seed)

    # equity: GBM using *pathwise* rates
    import numpy as np
    n_steps = int(maturity_years * steps_per_year)
    dt = maturity_years / n_steps
    sqrt_dt = np.sqrt(dt)

    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("For antithetic, n_paths must be even.")
        half = n_paths // 2
        Z_half = rng.standard_normal(size=(half, n_steps))
        Z = np.vstack([Z_half, -Z_half])
    else:
        Z = rng.standard_normal(size=(n_paths, n_steps))

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = s0

    for k in range(n_steps):
        r_k = r_paths[:, k]
        dW = Z[:, k] * sqrt_dt
        S_t = S[:, k]
        dlogS = (r_k - 0.5 * sigma ** 2) * dt + sigma * dW
        S[:, k + 1] = S_t * np.exp(dlogS)

    price, se = price_autocall_athena_from_paths(
        S_paths=S,
        df_paths=df,
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_per_year,
        maturity_years=maturity_years,
        obs_freq=obs_freq,
        call_barrier=1.0,
        protection_barrier=0.6,
        steps_per_year=steps_per_year,
    )
    return price, se


def price_autocall_with_hybrid_model(
    nominal, strike, coupon_per_year,
    maturity_years, obs_freq,
    s0, v0, r0,
    kappa, theta, xi, rho_sv,
    a, b, sigma_r, rho_sr, rho_vr,
    steps_per_year, n_paths,
    seed=None, antithetic=False,
):
    hybrid = HestonHullWhiteHybrid(
        s0=s0, v0=v0, r0=r0,
        kappa=kappa, theta=theta, xi=xi, rho_sv=rho_sv,
        a=a, b=b, sigma_r=sigma_r,
        rho_sr=rho_sr, rho_vr=rho_vr,
        steps_per_year=steps_per_year, n_paths=n_paths, seed=seed,
    )
    S, v, r, df = hybrid.simulate_paths(maturity_years, antithetic=antithetic)

    price, se = price_autocall_athena_from_paths(
        S_paths=S,
        df_paths=df,
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_per_year,
        maturity_years=maturity_years,
        obs_freq=obs_freq,
        call_barrier=1.0,
        protection_barrier=0.6,
        steps_per_year=steps_per_year,
    )
    return price, se
