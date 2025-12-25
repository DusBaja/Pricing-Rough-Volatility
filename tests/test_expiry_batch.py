import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from src.pricer.market.surface import parse_sx5e_bbg_surface
from src.pricer.core.models import Model
from src.pricer.vanilla_mc.vanilla_mc_surface import mc_prices_for_expiry
from src.pricer.market.black import implied_vol_black
from src.pricer.market.xi0_builder import build_xi0_from_atm

surf = parse_sx5e_bbg_surface("data/raw/OptionsSX5E.xlsx", val_date="2025-12-15")

# the longest expiry
expiry = surf["expiry"].max()
q = surf[surf["expiry"] == expiry].copy()
q = q[np.isfinite(q["F_bbg"])].copy()

T = float(q["T"].iloc[0])
F = float(q["F_bbg"].iloc[0])
xi0_times, xi0_values = build_xi0_from_atm(surf)
print("xi0_times:", xi0_times)
print("xi0_values (var):", xi0_values)
print("xi0 vols:", np.sqrt(xi0_values))
m = Model(
    spot_process="GBM",
    rate_process="FLAT",
    vol_process="RBERGOMI",
    s0=F,
    sigma=0.14,
    r0=0.02,
    rough_H=0.10,
    rough_nu=1.5,
    rough_rho=-0.7,
    xi0_times=xi0_times,
    xi0_values=xi0_values,
    n_paths=200000,
    steps_per_year=128,
    seed=12345
)
test_times = np.array([0.01, T, float(xi0_times.max())], dtype=float)
print("xi0_at_times:", test_times, m._xi0_at_times(test_times))


q2 = mc_prices_for_expiry(m, T, q, antithetic=True)

dfT = float(m._df_at_times(np.array([T]))[0])

for i in range(min(10, len(q2))):
    K = float(q2["K"].iloc[i])
    cp = q2["cp"].iloc[i]
    iv_mkt = float(q2["iv"].iloc[i])
    price = float(q2["price_mc"].iloc[i])
    se = float(q2["se_mc"].iloc[i])
    iv_model = implied_vol_black(F, K, T, dfT, price, cp)
    print(f"K={K:.1f} {cp}  IV_mkt={iv_mkt:.4f}  IV_model={iv_model:.4f}  SE(price)={se:.4f}")
