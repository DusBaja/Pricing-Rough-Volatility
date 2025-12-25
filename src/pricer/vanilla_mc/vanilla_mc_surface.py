import numpy as np
import pandas as pd
from typing import Dict
from pricer.core.models import Model

def mc_prices_for_expiry(model: Model, T: float, quotes_T: pd.DataFrame, antithetic: bool = True) -> pd.DataFrame:
    """
    quotes_T: dataframe for a single expiry T with columns ['K','cp', ...]
    Returns a copy with added columns: ['price_mc','se_mc'] using ONE simulation for all strikes.
    """
    paths = model.simulate_paths(maturity_years=T, antithetic=antithetic)
    ST = paths["S"][:, -1]
    dfT = paths["df"][:, -1]

    out = quotes_T.copy()
    prices = []
    ses = []

    for row in out.itertuples(index=False):
        K = float(row.K)
        cp = row.cp

        if cp == "C":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)

        pv = dfT * payoff

        
        pv = pv[np.isfinite(pv)]

        price = float(np.mean(pv))
        se = float(np.std(pv, ddof=1) / np.sqrt(len(pv)))

        prices.append(price)
        ses.append(se)

    out["price_mc"] = prices
    out["se_mc"] = ses
    return out
