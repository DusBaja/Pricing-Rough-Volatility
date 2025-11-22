# main.py
from mc_pricer import (
    price_autocall_mc_classic,
    price_autocall_with_vol_model,
    price_autocall_with_rate_model,
    price_autocall_with_hybrid_model,
)
from sensitivities import all_param_sensitivities

if __name__ == "__main__":
    # Product setup
    nominal = 100.0
    s0 = 100.0
    strike = 100.0

    coupon_pa = 0.08          # 8% p.a.
    maturity = 5.0            # 5 years
    obs_freq = "annual"       # could be "monthly"/"daily"

    # MC setup
    steps_per_year = 252
    n_paths = 40_000
    seed = 1234
    antithetic = True

    # Flat rate / vol
    r0 = 0.02
    sigma_bs = 0.20

    # Heston params (for vol model and hybrid)
    v0 = 0.04
    kappa = 1.5
    theta = 0.04
    xi_heston = 0.5
    rho_sv = -0.7

    # Hull–White params (for rate model and hybrid)
    a = 0.05
    b = 0.02
    sigma_r = 0.01

    # Correlations with rate in hybrid
    rho_sr = 0.0
    rho_vr = 0.0

    print("=== Autocall Athena Monte Carlo Pricing ===\n")

    # 1) Plain MC classic (GBM + flat rate)
    price_mc, se_mc = price_autocall_mc_classic(
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_pa,
        maturity_years=maturity,
        obs_freq=obs_freq,
        s0=s0,
        r0=r0,
        sigma=sigma_bs,
        steps_per_year=steps_per_year,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    print(f"[1] Plain MC (GBM, flat r)          : {price_mc:8.4f}  (SE = {se_mc:7.4f})")

    # 2) Vol model only (Heston, flat rate)
    price_heston, se_heston = price_autocall_with_vol_model(
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_pa,
        maturity_years=maturity,
        obs_freq=obs_freq,
        s0=s0,
        v0=v0,
        r0=r0,
        kappa=kappa,
        theta=theta,
        xi=xi_heston,
        rho_sv=rho_sv,
        steps_per_year=steps_per_year,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    print(f"[2] Vol model only (Heston, flat r) : {price_heston:8.4f}  (SE = {se_heston:7.4f})")

    # 3) Rate model only (Hull–White, const vol)
    price_hw, se_hw = price_autocall_with_rate_model(
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_pa,
        maturity_years=maturity,
        obs_freq=obs_freq,
        s0=s0,
        sigma=sigma_bs,
        r0=r0,
        a=a,
        b=b,
        sigma_r=sigma_r,
        steps_per_year=steps_per_year,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    print(f"[3] Rate model only (GBM + HW)      : {price_hw:8.4f}  (SE = {se_hw:7.4f})")

    # 4) Vol + rate model (Heston + Hull–White)
    price_hyb, se_hyb = price_autocall_with_hybrid_model(
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_pa,
        maturity_years=maturity,
        obs_freq=obs_freq,
        s0=s0,
        v0=v0,
        r0=r0,
        kappa=kappa,
        theta=theta,
        xi=xi_heston,
        rho_sv=rho_sv,
        a=a,
        b=b,
        sigma_r=sigma_r,
        rho_sr=rho_sr,
        rho_vr=rho_vr,
        steps_per_year=steps_per_year,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    print(f"[4] Hybrid (Heston + HW)            : {price_hyb:8.4f}  (SE = {se_hyb:7.4f})")

    # ===============================
    # Sensitivities for chosen model
    # ===============================
    print("\n=== Sensitivities for chosen model: Heston + Hull–White ===")
    greek_n_paths = 5_000
    chosen_params = dict(
        nominal=nominal,
        strike=strike,
        coupon_per_year=coupon_pa,
        maturity_years=maturity,
        obs_freq=obs_freq,
        s0=s0,
        v0=v0,
        r0=r0,
        kappa=kappa,
        theta=theta,
        xi=xi_heston,
        rho_sv=rho_sv,
        a=a,
        b=b,
        sigma_r=sigma_r,
        rho_sr=rho_sr,
        rho_vr=rho_vr,
        steps_per_year=steps_per_year,
        n_paths=greek_n_paths,
        seed=seed,
        antithetic=antithetic,
    )

    # All parameters of the model you want sensitivities for:
    param_names = [
        "s0", "v0", "r0",
        "kappa", "theta", "xi", "rho_sv",
        "a", "b", "sigma_r",
        "rho_sr", "rho_vr",
    ]

    sens = all_param_sensitivities(price_autocall_with_hybrid_model, chosen_params, param_names, rel_bump=0.01,central=False)

    for name in param_names:
        print(f"dPrice/d{name:7s} = {sens[name]: .6f}")
