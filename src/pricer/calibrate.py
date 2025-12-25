# Unified vanilla calibration script for your project.
#
# Supports: rbergomi, heston, rough_fbm, rfsv, heston_hw (calibrate equity part; HW params fixed)
#
# Key conventions:
# - Calibrate on SX5E Bloomberg surface (OptionsSX5E.xlsx) using near-ATM / OTM quotes.
# - "Forward trick" for vanilla calibration: set model.s0 = F_bbg per expiry and simulate with r0=0
#   so S_t is (approximately) driftless; discount with DF(0,T) from your curve.
# - Objective: vega-weighted MSE of implied vols, with one simulation per expiry (shared paths).
#
# Usage examples:
#   python calibrate.py --model rbergomi --npaths_obj 150000 --npaths_report 300000
#   python calibrate.py --model heston --exclude_shortest --npaths_obj 150000 --npaths_report 300000
#   python calibrate.py --model rfsv --npaths_obj 150000 --npaths_report 300000
#   python calibrate.py --model rough_fbm --npaths_obj 150000 --npaths_report 300000

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.optimize as opt


from pricer.core.models import Model
from pricer.market.surface import parse_sx5e_bbg_surface
from pricer.market.xi0_builder import build_xi0_from_atm
from pricer.vanilla_mc.vanilla_mc import mc_price_vanilla
from pricer.vanilla_mc.vanilla_mc_surface import mc_prices_for_expiry
from pricer.market.black import implied_vol_black



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

    # de-duplicate: keep last occurrence for each time
    _, idx_first_rev = np.unique(disc_times[::-1], return_index=True)
    idx_last = len(disc_times) - 1 - idx_first_rev
    idx_last = np.sort(idx_last)
    disc_times = disc_times[idx_last]
    disc_rates = disc_rates[idx_last]
    return disc_times, disc_rates


def vega_proxy(F: float, K: float, T: float, iv: float) -> float:
    """Black vega proxy (ignores DF). Used only for weighting errors."""
    if T <= 0.0 or iv <= 0.0 or F <= 0.0 or K <= 0.0:
        return 0.0
    srt = iv * math.sqrt(T)
    if srt <= 1e-12:
        return 0.0
    d1 = (math.log(F / K) + 0.5 * iv * iv * T) / srt
    phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return F * phi * math.sqrt(T)


def add_logm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["logm"] = np.log(out["K"] / out["F_bbg"])
    return out


def filter_otm(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only OTM options (calls K>=F, puts K<=F)."""
    return df[
        ((df["cp"] == "C") & (df["K"] >= df["F_bbg"])) |
        ((df["cp"] == "P") & (df["K"] <= df["F_bbg"]))
    ].copy()


def maybe_exclude_shortest(df: pd.DataFrame, exclude_shortest: bool) -> pd.DataFrame:
    if not exclude_shortest:
        return df
    tmin = float(df["T"].min())
    return df[df["T"] > tmin].copy()


def group_objective_iv_mse(
    model: Model,
    surf_otm: pd.DataFrame,
    n_paths: int,
    steps_per_year: int,
    seed: int,
) -> float:
    """
    Generic objective: vega-weighted IV MSE.
    Assumes:
      - surf_otm has columns: expiry, T, F_bbg, K, cp, iv
      - model.s0 will be overwritten per expiry to F_bbg ("forward trick")
    """
    model.n_paths = int(n_paths)
    model.steps_per_year = int(steps_per_year)
    model.seed = int(seed)

    num = 0.0
    den = 0.0

    for _, q in surf_otm.groupby("expiry"):
        T = float(q["T"].iloc[0])
        Fb = q["F_bbg"].dropna()
        if len(Fb) == 0:
            continue
        F = float(Fb.iloc[0])

        # Forward trick: simulate with s0 = forward
        model.s0 = F
        dfT = float(model._df_at_times(np.array([T], dtype=float))[0])

        q_mc = mc_prices_for_expiry(model, T, q, antithetic=True)

        for row in q_mc.itertuples(index=False):
            K = float(row.K)
            cp = row.cp
            iv_mkt = float(row.iv)
            price = float(row.price_mc)

            iv_model = implied_vol_black(F, K, T, dfT, price, cp)

            w = vega_proxy(F, K, T, iv_mkt)
            if not np.isfinite(w) or w <= 0.0:
                continue

            err = iv_model - iv_mkt
            num += w * (err * err)
            den += w

    return num / max(den, 1e-16)


def report_by_expiry(
    model: Model,
    surf_otm: pd.DataFrame,
    n_paths: int,
    steps_per_year: int,
    seed: int,
    header: str,
) -> None:
    model.n_paths = int(n_paths)
    model.steps_per_year = int(steps_per_year)
    model.seed = int(seed)

    print("\n" + header)

    for expiry, q in surf_otm.groupby("expiry"):
        T = float(q["T"].iloc[0])
        F = float(q["F_bbg"].dropna().iloc[0])

        model.s0 = F
        dfT = float(model._df_at_times(np.array([T], dtype=float))[0])

        q_mc = mc_prices_for_expiry(model, T, q, antithetic=True)

        iv_model = []
        for row in q_mc.itertuples(index=False):
            iv_model.append(
                implied_vol_black(F, float(row.K), T, dfT, float(row.price_mc), row.cp)
            )

        iv_model = np.array(iv_model, dtype=float)
        iv_mkt = q["iv"].to_numpy(dtype=float)
        diff = iv_model - iv_mkt

        rmse = float(np.sqrt(np.mean(diff * diff)))
        print(
            "Expiry", expiry,
            "T", round(T, 6),
            "RMSE_IV", rmse,
            "min_err", float(np.min(diff)),
            "max_err", float(np.max(diff)),
            "n", len(q),
        )


# -----------------------------
# Calibrator specs
# -----------------------------
H_FIXED_DEFAULT = 0.10


@dataclass
class CalibResult:
    model: str
    cut: float
    fun: float
    params: Dict[str, float]
    meta: Dict[str, float]


# -----------------------------
# Model builders (for calibration)
# -----------------------------
def build_base_for_calibration(
    disc_times: np.ndarray,
    disc_rates: np.ndarray,
    xi0_times: np.ndarray,
    xi0_values: np.ndarray,
    steps_per_year: int,
    n_paths: int,
    seed: int,
) -> Model:
    # This base object provides curve + xi0; we override processes per model.
    return Model(
        spot_process="GBM",
        rate_process="FLAT",
        vol_process="FLAT",
        s0=1.0,          # overwritten per expiry
        sigma=0.2,
        r0=0.0,          # IMPORTANT for forward-trick driftless simulation
        disc_times=disc_times,
        disc_rates=disc_rates,
        xi0_times=xi0_times,
        xi0_values=xi0_values,
        n_paths=n_paths,
        steps_per_year=steps_per_year,
        seed=seed,
    )


def build_model_rbergomi(base: Model, eta: float, rho: float, H: float) -> Model:
    return Model(
        spot_process="GBM",
        rate_process="FLAT",
        vol_process="RBERGOMI",
        s0=base.s0,
        sigma=base.sigma,
        r0=0.0,  # driftless
        rough_H=H,
        rough_nu=eta,
        rough_rho=rho,
        xi0_times=base.xi0_times,
        xi0_values=base.xi0_values,
        disc_times=base.disc_times,
        disc_rates=base.disc_rates,
        n_paths=base.n_paths,
        steps_per_year=base.steps_per_year,
        seed=base.seed,
    )


def build_model_rough_fbm(base: Model, nu: float, H: float, m: float = 0.0) -> Model:
    return Model(
        spot_process="GBM",
        rate_process="FLAT",
        vol_process="ROUGH_FBM",
        s0=base.s0,
        sigma=base.sigma,
        r0=0.0,
        rough_H=H,
        rough_nu=nu,
        rough_m=m,
        disc_times=base.disc_times,
        disc_rates=base.disc_rates,
        n_paths=base.n_paths,
        steps_per_year=base.steps_per_year,
        seed=base.seed,
    )


def build_model_rfsv(base: Model, nu: float, alpha: float, H: float, m: float = 0.0) -> Model:
    return Model(
        spot_process="GBM",
        rate_process="FLAT",
        vol_process="RFSV",
        s0=base.s0,
        sigma=base.sigma,
        r0=0.0,
        rough_H=H,
        rough_nu=nu,
        rough_alpha=alpha,
        rough_m=m,
        disc_times=base.disc_times,
        disc_rates=base.disc_rates,
        n_paths=base.n_paths,
        steps_per_year=base.steps_per_year,
        seed=base.seed,
    )


def build_model_heston(base: Model, v0: float, kappa: float, theta: float, xi: float, rho_sv: float) -> Model:
    return Model(
        spot_process="HESTON",
        rate_process="FLAT",
        vol_process="FLAT",
        s0=base.s0,
        r0=0.0,
        disc_times=base.disc_times,
        disc_rates=base.disc_rates,
        steps_per_year=base.steps_per_year,
        n_paths=base.n_paths,
        seed=base.seed,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho_sv=rho_sv,
    )


# -----------------------------
# Objective wrappers per model
# -----------------------------
def objective_rbergomi(x, surf, base, H_fixed, n_paths, steps_per_year, seed) -> float:
    eta, rho = float(x[0]), float(x[1])
    if not (0.02 < float(H_fixed) < 0.45):
        return 1e6
    if not (1e-4 < eta < 5.0):
        return 1e6
    if not (-0.999 < rho < -1e-6):
        return 1e6
    m = build_model_rbergomi(base, eta=eta, rho=rho, H=float(H_fixed))
    return group_objective_iv_mse(m, surf, n_paths=n_paths, steps_per_year=steps_per_year, seed=seed)


def objective_heston(x, surf, base, kappa_fixed, theta_fixed, n_paths, steps_per_year, seed) -> float:
    v0, xi, rho_sv = float(x[0]), float(x[1]), float(x[2])
    if not (1e-8 < v0 < 0.75):
        return 1e6
    if not (1e-4 < xi < 10.0):
        return 1e6
    if not (-0.999 < rho_sv < -1e-6):
        return 1e6

    m = build_model_heston(base, v0=v0, kappa=float(kappa_fixed), theta=float(theta_fixed), xi=xi, rho_sv=rho_sv)
    mse = group_objective_iv_mse(m, surf, n_paths=n_paths, steps_per_year=steps_per_year, seed=seed)

    lam_rho = 5e-4
    rho_soft = 0.95
    pen_rho = lam_rho * max(0.0, abs(rho_sv) - rho_soft) ** 2

    lam_v0 = 1e-3
    pen_v0 = lam_v0 * (v0 - float(theta_fixed)) ** 2

    return mse + pen_rho + pen_v0


def objective_rough_fbm(x, surf, base, H_fixed, n_paths, steps_per_year, seed) -> float:
    nu = float(x[0])
    if not (1e-4 < nu < 5.0):
        return 1e6
    m = build_model_rough_fbm(base, nu=nu, H=float(H_fixed), m=0.0)
    return group_objective_iv_mse(m, surf, n_paths=n_paths, steps_per_year=steps_per_year, seed=seed)


def objective_rfsv(x, surf, base, H_fixed, n_paths, steps_per_year, seed) -> float:
    nu, alpha = float(x[0]), float(x[1])
    if not (1e-4 < nu < 5.0):
        return 1e6
    if not (1e-4 < alpha < 20.0):
        return 1e6
    m = build_model_rfsv(base, nu=nu, alpha=alpha, H=float(H_fixed), m=0.0)
    return group_objective_iv_mse(m, surf, n_paths=n_paths, steps_per_year=steps_per_year, seed=seed)


# -----------------------------
# Calibration runner
# -----------------------------
def run_calibration(
    model_name: str,
    surf_otm: pd.DataFrame,
    base: Model,
    cut_list: np.ndarray,
    H_fixed: float,
    npaths_obj: int,
    npaths_report: int,
    steps_per_year: int,
    seed: int,
    exclude_shortest: bool,
    kappa_fixed: float,
    theta_fixed: Optional[float],
) -> CalibResult:
    # Apply exclude_shortest consistently
    surf_work = maybe_exclude_shortest(surf_otm.copy(), exclude_shortest=exclude_shortest)

    best: Optional[Tuple[float, float, np.ndarray]] = None  # (cut, fun, x)
    last_x: Optional[np.ndarray] = None

    for cut in cut_list:
        s = add_logm(surf_work)
        s = s[s["logm"].abs() <= float(cut)].copy()
        if len(s) == 0:
            continue

        if model_name == "rbergomi":
            if last_x is None:
                last_x = np.array([0.05, -0.55], dtype=float)
            bounds = [(1e-4, 2.5), (-0.95, -1e-4)]
            obj0 = objective_rbergomi(last_x, s, base, H_fixed, npaths_obj, steps_per_year, seed)
            print(f"\nCUT {cut:.2f} objective@start={obj0:.8f} start x={last_x}")

            res = opt.minimize(
                lambda p: objective_rbergomi(p, s, base, H_fixed, npaths_obj, steps_per_year, seed),
                x0=last_x,
                bounds=bounds,
                method="L-BFGS-B",
                options=dict(maxiter=25),
            )

        elif model_name in ("heston", "heston_hw"):
            if theta_fixed is None:
                theta_fixed = float(base.xi0_values[-1])
            if last_x is None:
                last_x = np.array([theta_fixed, 1.2, -0.75], dtype=float)

            bounds = [(1e-8, 0.75), (1e-4, 5.0), (-0.95, -1e-4)]
            obj0 = objective_heston(last_x, s, base, kappa_fixed, float(theta_fixed), npaths_obj, steps_per_year, seed)
            print(f"\nCUT {cut:.2f} objective@start={obj0:.8f} start x={last_x}")

            res = opt.minimize(
                lambda p: objective_heston(p, s, base, kappa_fixed, float(theta_fixed), npaths_obj, steps_per_year, seed),
                x0=last_x,
                bounds=bounds,
                method="L-BFGS-B",
                options=dict(maxiter=25),
            )

        elif model_name == "rough_fbm":
            if last_x is None:
                last_x = np.array([0.5], dtype=float)
            bounds = [(1e-4, 5.0)]
            obj0 = objective_rough_fbm(last_x, s, base, H_fixed, npaths_obj, steps_per_year, seed)
            print(f"\nCUT {cut:.2f} objective@start={obj0:.8f} start x={last_x}")

            res = opt.minimize(
                lambda p: objective_rough_fbm(p, s, base, H_fixed, npaths_obj, steps_per_year, seed),
                x0=last_x,
                bounds=bounds,
                method="L-BFGS-B",
                options=dict(maxiter=25),
            )

        elif model_name == "rfsv":
            if last_x is None:
                last_x = np.array([0.5, 2.0], dtype=float)  # nu, alpha
            bounds = [(1e-4, 5.0), (1e-3, 20.0)]
            obj0 = objective_rfsv(last_x, s, base, H_fixed, npaths_obj, steps_per_year, seed)
            print(f"\nCUT {cut:.2f} objective@start={obj0:.8f} start x={last_x}")

            res = opt.minimize(
                lambda p: objective_rfsv(p, s, base, H_fixed, npaths_obj, steps_per_year, seed),
                x0=last_x,
                bounds=bounds,
                method="L-BFGS-B",
                options=dict(maxiter=25),
            )

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        print("Result:", res.message)
        print("  fun:", float(res.fun))
        print("  x  :", res.x)

        last_x = res.x.copy()
        if best is None or float(res.fun) < best[1]:
            best = (float(cut), float(res.fun), res.x.copy())

    if best is None:
        raise RuntimeError("Calibration produced no valid result (best is None).")

    best_cut, best_fun, best_x = best
    print(f"\nBEST (cut={best_cut:.2f}) fun={best_fun:.8f} x={best_x}")

    # Report set must match calibration set rules (cut + exclude_shortest)
    surf_report = add_logm(maybe_exclude_shortest(surf_otm.copy(), exclude_shortest=exclude_shortest))
    surf_report = surf_report[surf_report["logm"].abs() <= best_cut].copy()

    if model_name == "rbergomi":
        eta, rho = float(best_x[0]), float(best_x[1])
        m = build_model_rbergomi(base, eta=eta, rho=rho, H=H_fixed)
        report_by_expiry(
            m, surf_report, n_paths=npaths_report, steps_per_year=steps_per_year, seed=seed,
            header=f"CALIBRATED rBergomi: H={H_fixed:.4f} eta={eta:.6f} rho={rho:.6f}"
        )
        return CalibResult(
            model=model_name, cut=best_cut, fun=best_fun,
            params=dict(H=float(H_fixed), eta=eta, rho=rho),
            meta=dict(exclude_shortest=float(exclude_shortest)),
        )

    if model_name in ("heston", "heston_hw"):
        if theta_fixed is None:
            theta_fixed = float(base.xi0_values[-1])
        v0, xi, rho_sv = float(best_x[0]), float(best_x[1]), float(best_x[2])
        m = build_model_heston(base, v0=v0, kappa=kappa_fixed, theta=float(theta_fixed), xi=xi, rho_sv=rho_sv)
        report_by_expiry(
            m, surf_report, n_paths=npaths_report, steps_per_year=steps_per_year, seed=seed,
            header=f"CALIBRATED Heston: kappa={kappa_fixed:.4f} theta={float(theta_fixed):.6f} "
                   f"v0={v0:.6f} xi={xi:.6f} rho_sv={rho_sv:.6f}"
        )
        return CalibResult(
            model=model_name, cut=best_cut, fun=best_fun,
            params=dict(kappa=float(kappa_fixed), theta=float(theta_fixed), v0=v0, xi=xi, rho_sv=rho_sv),
            meta=dict(exclude_shortest=float(exclude_shortest)),
        )

    if model_name == "rough_fbm":
        nu = float(best_x[0])
        m = build_model_rough_fbm(base, nu=nu, H=H_fixed, m=0.0)
        report_by_expiry(
            m, surf_report, n_paths=npaths_report, steps_per_year=steps_per_year, seed=seed,
            header=f"CALIBRATED ROUGH_FBM: H={H_fixed:.4f} nu={nu:.6f} (m fixed to 0)"
        )
        return CalibResult(
            model=model_name, cut=best_cut, fun=best_fun,
            params=dict(H=float(H_fixed), nu=nu, m=0.0),
            meta=dict(exclude_shortest=float(exclude_shortest)),
        )

    if model_name == "rfsv":
        nu, alpha = float(best_x[0]), float(best_x[1])
        m = build_model_rfsv(base, nu=nu, alpha=alpha, H=H_fixed, m=0.0)
        report_by_expiry(
            m, surf_report, n_paths=npaths_report, steps_per_year=steps_per_year, seed=seed,
            header=f"CALIBRATED RFSV: H={H_fixed:.4f} nu={nu:.6f} alpha={alpha:.6f} (m fixed to 0)"
        )
        return CalibResult(
            model=model_name, cut=best_cut, fun=best_fun,
            params=dict(H=float(H_fixed), nu=nu, alpha=alpha, m=0.0),
            meta=dict(exclude_shortest=float(exclude_shortest)),
        )

    raise RuntimeError("Unreachable.")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["rbergomi", "heston", "rough_fbm", "rfsv", "heston_hw"])
    parser.add_argument("--surface", type=str, default="data/raw/OptionsSX5E.xlsx")
    parser.add_argument("--curve", type=str, default="data/raw/EURvs3MEuribor-Funding.xlsx")
    parser.add_argument("--val_date", type=str, default="2025-12-15")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps_per_year", type=int, default=128)

    parser.add_argument("--npaths_obj", type=int, default=50000)
    parser.add_argument("--npaths_report", type=int, default=100000)

    parser.add_argument("--cut_min", type=float, default=0.02)
    parser.add_argument("--cut_max", type=float, default=0.05)
    parser.add_argument("--cut_step", type=float, default=0.01)

    parser.add_argument("--H", type=float, default=H_FIXED_DEFAULT)
    parser.add_argument("--exclude_shortest", action="store_true",
                        help="Exclude shortest expiry from calibration set (recommended for Heston on 4D).")

    # Heston hyperparams
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--theta", type=float, default=None,
                        help="If set, override theta fixed; else uses base xi0 last value.")

    parser.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Write best params to JSON. If empty, defaults to calib_<model>.json in current folder."
    )

    args = parser.parse_args()

    disc_times, disc_rates = load_funding_curve(args.curve)

   
    surf_all = parse_sx5e_bbg_surface(args.surface, val_date=args.val_date)
    surf_all = surf_all.dropna(subset=["F_bbg", "iv", "T", "K", "cp"]).copy()

    # Build xi0 from near-ATM points
    xi0_times, xi0_values = build_xi0_from_atm(surf_all)

    # OTM set
    surf_otm = filter_otm(surf_all)

    # Base model (curve + xi0 holder)
    base = build_base_for_calibration(
        disc_times=disc_times,
        disc_rates=disc_rates,
        xi0_times=xi0_times,
        xi0_values=xi0_values,
        steps_per_year=args.steps_per_year,
        n_paths=20000,
        seed=args.seed,
    )

    # Quick sanity check (one quote)
    row = surf_otm.iloc[0]
    T = float(row["T"])
    K = float(row["K"])
    cp = row["cp"]
    F = float(row["F_bbg"])
    iv_mkt = float(row["iv"])
    base.s0 = F
    base.sigma = 0.14

    price, se = mc_price_vanilla(base, T=T, K=K, cp=cp)
    dfT = float(base._df_at_times(np.array([T], dtype=float))[0])
    iv_model = implied_vol_black(F, K, T, dfT, price, cp)
    print("Market IV:", iv_mkt)
    print("Model  IV:", iv_model)
    print("Price SE :", se)

    cuts = np.arange(args.cut_min, args.cut_max + 0.5 * args.cut_step, args.cut_step)

    theta_fixed = None if args.theta is None else float(args.theta)

    res = run_calibration(
        model_name=args.model,
        surf_otm=surf_otm,
        base=base,
        cut_list=cuts,
        H_fixed=float(args.H),
        npaths_obj=int(args.npaths_obj),
        npaths_report=int(args.npaths_report),
        steps_per_year=int(args.steps_per_year),
        seed=int(args.seed),
        exclude_shortest=bool(args.exclude_shortest),
        kappa_fixed=float(args.kappa),
        theta_fixed=theta_fixed,
    )

    out_json = args.out_json.strip()
    if out_json == "":
        out_json = f"calib_{args.model}.json"

    payload = {
        "model": res.model,
        "cut": res.cut,
        "fun": res.fun,
        "params": res.params,
        "meta": res.meta,
        "val_date": args.val_date,
        "surface": args.surface,
        "curve": args.curve,
        "seed": int(args.seed),
        "steps_per_year": int(args.steps_per_year),
        "npaths_obj": int(args.npaths_obj),
        "npaths_report": int(args.npaths_report),
    }

    
    if getattr(base, "xi0_times", None) is not None and getattr(base, "xi0_values", None) is not None:
        payload["xi0_times"] = [float(x) for x in np.asarray(base.xi0_times).ravel()]
        payload["xi0_values"] = [float(x) for x in np.asarray(base.xi0_values).ravel()]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote calibration to {out_json}")


if __name__ == "__main__":
    main()