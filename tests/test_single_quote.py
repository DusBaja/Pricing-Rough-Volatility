import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from src.pricer.market.surface import parse_sx5e_bbg_surface
from src.pricer.core.models import Model

import src.pricer.core.models as models
from src.pricer.vanilla_mc.vanilla_mc import mc_price_vanilla
from src.pricer.market.black import implied_vol_black

surf = parse_sx5e_bbg_surface("data/raw/OptionsSX5E.xlsx", val_date="2025-12-15")

row = surf[surf["expiry"] == surf["expiry"].max()].iloc[0]
T = float(row["T"])
K = float(row["K"])
iv_mkt = float(row["iv"])
cp = row["cp"]
F = float(row["F_bbg"])


m = Model(
    spot_process="GBM", rate_process="FLAT", vol_process="RBERGOMI",
    s0=F, sigma=0.14, r0=0.02,
    rough_H=0.10, rough_nu=1.5, rough_rho=-0.7,
    n_paths=50000, steps_per_year=128, seed=12345
)

print("Using models from:", models.__file__)

paths = m.simulate_paths(maturity_years=T, antithetic=True)
ST = paths["S"][:, -1]
vT = paths["v"][:, -1]

print("ST finite %:", np.isfinite(ST).mean(), "min/max:", np.nanmin(ST), np.nanmax(ST))
print("vT finite %:", np.isfinite(vT).mean(), "min/max:", np.nanmin(vT), np.nanmax(vT))

dfT = float(m._df_at_times(np.array([T]))[0])
price, se = mc_price_vanilla(m, T, K, cp)
iv_model = implied_vol_black(F, K, T, dfT, price, cp)

print("Market IV:", iv_mkt, "Model IV:", iv_model, "SE(price):", se)
