import numpy as np
from typing import Tuple
from pricer.core.models import Model

def mc_price_vanilla(model: Model, T: float, K: float, cp: str, antithetic: bool = True) -> Tuple[float, float]:
    paths = model.simulate_paths(maturity_years=T, antithetic=antithetic)
    ST = paths["S"][:, -1]
    dfT = paths["df"][:, -1]

    if cp.upper() == "C":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    Y = dfT * payoff                     
    X = dfT * ST                          
    X0 = model.s0                         

    # beta = Cov(Y,X)/Var(X)
    Xc = X - X.mean()
    beta = 0.0 if np.var(X) < 1e-16 else np.mean((Y - Y.mean()) * Xc) / np.mean(Xc * Xc)

    Y_cv = Y - beta * (X - X0)

    price = float(np.mean(Y_cv))
    se = float(np.std(Y_cv, ddof=1) / np.sqrt(len(Y_cv)))
    return price, se
