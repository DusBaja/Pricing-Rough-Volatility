from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


from pricer.core.observation import ObservationFrequency
from pricer.core.models import Model
from pricer.core.products import AutocallAthenaProduct, AutocallPhoenixProduct
from pricer.core.pricing import MonteCarloPricer
from pricer.core.greeks import Greeks
from pricer.market.surface import parse_sx5e_bbg_surface
from pricer.market.xi0_builder import build_xi0_from_atm


def load_calibration(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_calib_path(model_name: str) -> str:
    return f"calib_{model_name}.json"

def tenor_to_years(tenor: str) -> float:
    t = str(tenor).strip().upper()
    t = t.split("_")[0]  # strip Bloomberg suffix
    m = re.fullmatch(r"(\d+)([MY])", t)
    if not m:
        raise ValueError(f"Unrecognized tenor after cleanup: {tenor}")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "M":
        return n / 12.0
    if unit == "Y":
        return float(n)
    raise ValueError(f"Unknown tenor unit: {tenor}")


def load_funding_curve(curve_path: str) -> Tuple[np.ndarray, np.ndarray]:
    c = pd.read_excel(curve_path, sheet_name="Worksheet")[["Tenor", "Yield"]].dropna()
    disc_times = np.array([tenor_to_years(x) for x in c["Tenor"].values], dtype=float)
    disc_rates = np.array(c["Yield"].values, dtype=float) / 100.0
    _, idx_first_rev = np.unique(disc_times[::-1], return_index=True)
    idx_last = len(disc_times) - 1 - idx_first_rev
    idx_last = np.sort(idx_last)
    disc_times = disc_times[idx_last]
    disc_rates = disc_rates[idx_last]
    return disc_times, disc_rates

def maybe_load_xi0_from_calib(calib: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if "xi0_times" in calib and "xi0_values" in calib:
        xt = np.array(calib["xi0_times"], dtype=float)
        xv = np.array(calib["xi0_values"], dtype=float)
        if len(xt) > 0 and len(xt) == len(xv):
            return xt, xv
    return None, None

def build_xi0_from_surface(surface_path: str, val_date: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebuild xi0 exactly like in calibration: load SX5E surface and call build_xi0_from_atm.
    """
    surf_all = parse_sx5e_bbg_surface(surface_path, val_date=val_date)
    surf_all = surf_all.dropna(subset=["F_bbg", "iv", "T", "K", "cp"]).copy()
    xi0_times, xi0_values = build_xi0_from_atm(surf_all)
    return np.asarray(xi0_times, dtype=float), np.asarray(xi0_values, dtype=float)


def build_model_from_calibration(
    model_name: str,
    calib: Dict[str, Any],
    *,
    s0: float,
    r0: float,
    sigma: float,
    disc_times: np.ndarray,
    disc_rates: np.ndarray,
    steps_per_year: int,
    n_paths: int,
    seed: int,
    # Allow passing xi0 ie the forward variance curve for rBergomi 
    xi0_times: Optional[np.ndarray] = None,
    xi0_values: Optional[np.ndarray] = None,
    hw_params: Optional[Dict[str, float]] = None,
) -> Model:
    """
    Build a pricing model using the saved calibration JSON.
    This is where we translate JSON -> Model(...) kwargs.

    IMPORTANT:
    - pricing is NOT "forward trick"; we use real spot s0 and real r0.
    - discounting uses disc_times/disc_rates always.
    """
    params = calib.get("params", {})
    # "gbm" is an alias for flat-vol GBM (Black–Scholes under RN).
    # It does not require a calibration JSON.
    if model_name.lower() == "gbm":
        return Model(
            spot_process="GBM",
            rate_process="FLAT",
            vol_process="FLAT",
            s0=float(s0),
            sigma=float(sigma),
            r0=float(r0),
            disc_times=disc_times,
            disc_rates=disc_rates,
            n_paths=int(n_paths),
            steps_per_year=int(steps_per_year),
            seed=int(seed),
        )

    #mname = calib.get("model", model_name)  # JSON should contain it; fallback to CLI choice
    mname = model_name #for hybrid bergomi rough vol 
    common = dict(
        s0=float(s0),
        r0=float(r0),
        sigma=float(sigma),
        disc_times=disc_times,
        disc_rates=disc_rates,
        steps_per_year=int(steps_per_year),
        n_paths=int(n_paths),
        seed=int(seed),
    )

    # --- rBergomi ---
    if mname.lower() == "rbergomi":
        if xi0_times is None or xi0_values is None:
            raise ValueError("rbergomi pricing needs xi0_times/xi0_values. Provide them (or load from your pipeline).")

        return Model(
            spot_process="GBM",
            rate_process="FLAT",
            vol_process="RBERGOMI",
            rough_H=float(params["H"]),
            rough_nu=float(params["eta"]),
            rough_rho=float(params["rho"]),
            xi0_times=xi0_times,
            xi0_values=xi0_values,
            **common,
        )
    if mname.lower() == "rbergomi_hw":
        if xi0_times is None or xi0_values is None:
            raise ValueError("rbergomi_hw needs xi0_times/xi0_values (load from calib JSON or rebuild from surface).")
        if hw_params is None:
            raise ValueError("rbergomi_hw needs hw_params (a, b, sigma_r, rho_sr, rho_vr).")

        return Model(
            spot_process="GBM",
            rate_process="HULLWHITE",
            vol_process="RBERGOMI",
            rough_H=float(params["H"]),
            rough_nu=float(params["eta"]),
            rough_rho=float(params["rho"]),
            xi0_times=xi0_times,
            xi0_values=xi0_values,
            a=float(hw_params["a"]),
            b=float(hw_params["b"]),
            sigma_r=float(hw_params["sigma_r"]),
            rho_sr=float(hw_params.get("rho_sr", 0.0)),
            rho_vr=float(hw_params.get("rho_vr", 0.0)),
            **common,
        )



    # --- Heston (flat rates) ---
    if mname.lower() == "heston":
        return Model(
            spot_process="HESTON",
            rate_process="FLAT",
            vol_process="FLAT",
            v0=float(params["v0"]),
            kappa=float(params["kappa"]),
            theta=float(params["theta"]),
            xi=float(params["xi"]),
            rho_sv=float(params["rho_sv"]),
            **common,
        )

    # --- Rough FBM (your simplified rough vol) ---
    if mname.lower() == "rough_fbm":
        return Model(
            spot_process="GBM",
            rate_process="FLAT",
            vol_process="ROUGH_FBM",
            rough_H=float(params["H"]),
            rough_nu=float(params["nu"]),
            rough_m=float(params.get("m", 0.0)),
            **common,
        )

    # --- RFSV (rough fractional stochastic vol) ---
    if mname.lower() == "rfsv":
        return Model(
            spot_process="GBM",
            rate_process="FLAT",
            vol_process="RFSV",
            rough_H=float(params["H"]),
            rough_nu=float(params["nu"]),
            rough_alpha=float(params["alpha"]),
            rough_m=float(params.get("m", 0.0)),
            **common,
        )

    # --- Heston + Hull-White pricing --- only calibrated on the vol surface (equity)
    if mname.lower() == "heston_hw":
        if hw_params is None:
            raise ValueError("heston_hw requires hw_params={'a':..,'b':..,'sigma_r':..,'rho_sr':..,'rho_vr':..} etc.")

        return Model(
            spot_process="HESTON",
            rate_process="HULLWHITE",
            vol_process="FLAT",
            v0=float(params["v0"]),
            kappa=float(params["kappa"]),
            theta=float(params["theta"]),
            xi=float(params["xi"]),
            rho_sv=float(params["rho_sv"]),
            a=float(hw_params["a"]),
            b=float(hw_params["b"]),
            sigma_r=float(hw_params["sigma_r"]),
            rho_sr=float(hw_params.get("rho_sr", 0.0)),
            rho_vr=float(hw_params.get("rho_vr", 0.0)),
            **common,
        )

    raise ValueError(f"Unsupported model in calibration JSON: {mname}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rbergomi",
                        choices=["gbm", "rbergomi", "heston", "rough_fbm", "rfsv", "heston_hw", "rbergomi_hw"])
    parser.add_argument("--curve", type=str, default="data/raw/EURvs3MEuribor-Funding.xlsx")
    parser.add_argument("--calib", type=str, default="",
                        help="Path to calib_*.json. If empty, uses calib_<model>.json")
    
    parser.add_argument("--surface", type=str, default="data/raw/OptionsSX5E.xlsx",
                        help="SX5E surface file used to rebuild xi0 for rBergomi (if xi0 not in calib json).")
    parser.add_argument("--val_date", type=str, default="2025-12-15",
                        help="Valuation date used by parse_sx5e_bbg_surface.")

    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--r0", type=float, default=0.02)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--steps_per_year", type=int, default=252)
    parser.add_argument("--n_paths", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=123)

    # HW params
    parser.add_argument("--hw_a", type=float, default=0.10)
    parser.add_argument("--hw_b", type=float, default=0.02)
    parser.add_argument("--hw_sigma_r", type=float, default=0.01)
    parser.add_argument("--hw_rho_sr", type=float, default=0.0)
    parser.add_argument("--hw_rho_vr", type=float, default=0.0)

    args = parser.parse_args()

    # The product speficiation 
    product = AutocallAthenaProduct(
        nominal=100.0,
        strike=100.0,
        coupon_per_year=0.04,
        maturity_years_=5.0,
        obs_freq=ObservationFrequency.ANNUAL,
        call_barrier=1.0,
        protection_barrier=0.7,
        steps_per_year=args.steps_per_year,
        with_memory=False,
    )

    # Our curves for discounting
    disc_times, disc_rates = load_funding_curve(args.curve)


    if args.model.lower() == "gbm":
        # GBM has no calibration
        calib = {"params": {}, "model": "gbm"}

    else:
        if args.calib is not None and args.calib.strip() != "":
            calib_path = args.calib.strip()
        else:
            calib_key = args.model.lower()

            # hybrids reuse equity calibration
            if calib_key == "heston_hw":
                calib_key = "heston"
            elif calib_key == "rbergomi_hw":
                calib_key = "rbergomi"

            calib_path = f"calib_{calib_key}.json"

        calib = load_calibration(calib_path)


    xi0_times: Optional[np.ndarray] = None
    xi0_values: Optional[np.ndarray] = None

    # xi0 is meaningful for rough-vol style models
    needs_xi0 = args.model in ("rbergomi", "rfsv", "rough_fbm","rbergomi_hw")
    if needs_xi0:
        xi0_times, xi0_values = maybe_load_xi0_from_calib(calib)

        if xi0_times is None or xi0_values is None:
            print(
                f"[{args.model}] xi0 not found in {calib_path}; "
                f"rebuilding from surface={args.surface} val_date={args.val_date}"
            )
            xi0_times, xi0_values = build_xi0_from_surface(args.surface, args.val_date)
    # Our model builder
    hw_params = None
    if args.model in ("heston_hw","rbergomi_hw"):
        hw_params = dict(
            a=args.hw_a,
            b=args.hw_b,
            sigma_r=args.hw_sigma_r,
            rho_sr=args.hw_rho_sr,
            rho_vr=args.hw_rho_vr,
        )

    model = build_model_from_calibration(
        args.model,
        calib,
        s0=args.s0,
        r0=args.r0,
        sigma=args.sigma,
        disc_times=disc_times,
        disc_rates=disc_rates,
        steps_per_year=args.steps_per_year,
        n_paths=args.n_paths,
        seed=args.seed,
        xi0_times=xi0_times,
        xi0_values=xi0_values,
        hw_params=hw_params,
    )

   
    print(f"Loaded calibration: {calib_path}")
    print(f"Pricing with model={model.vol_process} rate_process={model.rate_process} spot_process={model.spot_process}")

    pricer = MonteCarloPricer(antithetic=True, control_variate=True)#both methods. 
    greeks = Greeks(model, pricer)
    model.pricer = pricer
    model.greeks = greeks
    model.sensitivities = greeks  

    product.set_model(model)

  
    price, se = product.model.pricer.price(product)  # type: ignore[union-attr]
    print(f"Price = {price:.4f}, SE = {se:.4f}")

    g = product.model.greeks  # type: ignore[union-attr]

    model_name = getattr(model, "vol_process", "").upper()

    # Choose the "vol knob" for Vega/Vanna/Volga depending on the model
    if model_name == "RBERGOMI":
        vol_param = "xi0_level"
        vol_label = "xi0_level"
    elif model_name in ("RFSV", "ROUGH_FBM"):
        # Prefer rough_nu if available, else fall back to xi0_level if present
        if hasattr(model, "rough_nu"):
            vol_param = "rough_nu"
            vol_label = "rough_nu"
        elif getattr(model, "xi0_values", None) is not None:
            vol_param = "xi0_level"
            vol_label = "xi0_level"
        else:
            vol_param = "sigma"
            vol_label = "sigma"
    else:
        # Heston / Heston-HW: use v0 if present, else sigma
        if hasattr(model, "v0"):
            vol_param = "v0"
            vol_label = "v0"
        else:
            vol_param = "sigma"
            vol_label = "sigma"

    delta = g.delta(product)
    gamma = g.gamma(product)

    # Define Vega/Vanna/Volga wrt chosen vol_param (1% rel bumps)
    vega = g.vega(product, vol_param=vol_param, rel_bump=0.01)
    vanna = g.vanna(product, vol_param=vol_param, rel_bump=0.01)
    volga = g.volga(product, vol_param=vol_param, rel_bump=0.01)

    # Rho is DV01(1bp) if curve present, else flat-rate 1bp
    rho = g.rho(product)

    print(
        f"Delta={delta:.6f} Gamma={gamma:.6f} "
        f"Vega({vol_label} 1%)={vega:.6f} "
        f"Rho(DV01 1bp)={rho:.6f} "
        f"Vanna({vol_label} 1%)={vanna:.6f} "
        f"Volga({vol_label} 1%^2)={volga:.6f}"
    )

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
        if name == "disc_rates_parallel":
            print(f"curve_dv01(1bp) = {raw * 1e-4:.6f}")
        elif name == "r0":
            print(f"rho(bp) = {raw * 1e-4:.6f}")
        elif name in ("sigma", "xi", "v0", "xi0_level", "rough_nu"):
            print(f"vega(1%) wrt {name} = {raw * 0.01:.6f}")



if __name__ == "__main__":
    main()