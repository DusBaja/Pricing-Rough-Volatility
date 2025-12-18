# main.py
from __future__ import annotations

from observation import ObservationFrequency
from models import Model
from products import AutocallAthenaProduct, AutocallPhoenixProduct
from pricing import MonteCarloPricer
from greeks import Greeks


def main():
    product = AutocallAthenaProduct(
        nominal=100.0,
        strike=100.0,
        coupon_per_year=0.04,
        maturity_years_=5.0,
        obs_freq=ObservationFrequency.ANNUAL,
        call_barrier=1.0,
        protection_barrier=0.7,
        steps_per_year=252,
        with_memory=False
        
    )


    # Example: Rough full vol
    print("RFSV following parameters - without memory Athena :nominal=100.0, strike=100.0,coupon_per_year=0.04,maturity_years_=5.0,obs_freq=ObservationFrequency.ANNUAL,call_barrier=1.0,protection_barrier=0.7,steps_per_year=252,")
    model = Model(
    spot_process="GBM",
    rate_process="FLAT",
    vol_process="RFSV",
    s0=100.0,
    sigma=0.20,
    rough_H=0.10,
    rough_nu=0.30,
    rough_alpha=5e-4,   # experiment with this
    r0=0.02,
    steps_per_year=252,
    n_paths=20_000,
    seed=123,
)
    
    #model = Model(
    #spot_process="GBM",
    #rate_process="FLAT",
    #vol_process="ROUGH_FBM",
    #s0=100.0,
    #sigma=0.20,
    #rough_H=0.10,
    #rough_nu=0.30,
    #r0=0.02,
    #steps_per_year=252,
    #n_paths=20_000,
    #seed=42,)
    
    #model = Model(
    #    spot_process="HESTON",
    #    rate_process="HULLWHITE",
    #    s0=100.0,
    #    v0=0.04,
    #    r0=0.02,
    #    kappa=2.0,
    #    theta=0.04,
    #    xi=0.5,
    #    rho_sv=-0.6,
    #    a=0.1,
    #    b=0.02,
    #    sigma_r=0.01,
    #    rho_sr=0.3,
    #    rho_vr=0.2,
    #    steps_per_year=252,
    #    n_paths=20_000,
    #    seed=42,
    #)
    
    # Alternative examples:
    #model = Model(spot_process="GBM", rate_process="FLAT", s0=100, sigma=0.2, r0=0.02)
    #model = Model(spot_process="HESTON", rate_process="FLAT", ...)


    pricer = MonteCarloPricer(antithetic=False)
    greeks = Greeks(model, pricer)

    model.pricer = pricer
    model.greeks = greeks

    # Alias if you want a separate `.sensitivities`:
    model.sensitivities = greeks  # type: ignore[attr-defined]

    # Link product ↔ model so you get product.model.xxx
    product.set_model(model)

    # --------------------------------------------------------
    # 4) Pricing
    # --------------------------------------------------------
    price, se = product.model.pricer.price(product)  # type: ignore[union-attr]
    print(f"Price = {price:.4f}, SE = {se:.4f}")

    # or:
    # price2, se2 = model.price(product)

    # --------------------------------------------------------
    # 5) Greeks and parameter sensitivities
    # --------------------------------------------------------

    # Classic Greeks
    delta = product.model.greeks.delta(product)      # type: ignore[union-attr]
    gamma = product.model.greeks.gamma(product)      # type: ignore[union-attr]
    vega = product.model.greeks.vega(product)
    rho = product.model.greeks.rho(product)
    vanna =product.model.greeks.vanna(product)
    volga = product.model.greeks.volga(product)
    print(f"Delta = {delta:.6f}, Gamma = {gamma:.6f},Vega = {vega:.6f},Rho = {rho:.6f}, vanna = {vanna:.6f},volga = {volga:.6f}")


    
    param_names = model.default_sensitivity_parameters()
    all_sens = product.model.greeks.all_parameters(  # type: ignore[union-attr]
        product,
        param_names=param_names,
        rel_bump=0.01,
        central=False,
    )
    for name in param_names:
        raw = all_sens[name]
        print(f"dPrice/d{name:7s} = {raw: .6f}")

        if name == "r0":
            val = raw * 1e-4
            print(f"rho(bp) = {val:.6f}")
        elif name in ("sigma", "xi"):
            val = raw * 0.01
            print(f"vega(1%) wrt {name} = {val:.6f}")


if __name__ == "__main__":
    main()
