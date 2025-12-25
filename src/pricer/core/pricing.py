from __future__ import annotations

from typing import Tuple, Optional, Dict, Any

import numpy as np

from pricer.core.models import Model
from pricer.core.products import BaseProduct


class MonteCarloPricer:
    """
    Generic Monte Carlo pricer.

    It doesn() know the details of the model or product.
    It only assumes:
        - model.simulate_paths(maturity_years, antithetic)
        - product.payoff_from_paths(paths)

    Optional: supports common random numbers via rng_state ie having the same random seed generator to stabilize our outputs.
    """

    def __init__(self, antithetic: bool = False,control_variate: bool = False):
        self.antithetic = antithetic #method for variance reduction 
        self.control_variate = control_variate #another one implemented 
    
    def price(self,product: BaseProduct,model: Optional[Model] = None,rng_state: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Returns (price, standard_error).

        If `model` is None, it will try to use `product.model`.
        If `rng_state` is provided, resets model RNG to that state
        (common random numbers).
        """
        
        if model is None:
            if product.model is None:
                raise RuntimeError("No model provided to pricer, and product.model is None.")
            model = product.model

        
        if rng_state is not None:
            model.rng.bit_generator.state = rng_state

        paths = model.simulate_paths(
            maturity_years=product.maturity_years,
            antithetic=self.antithetic,
        )

        payoffs, dfs = product.payoff_from_paths(paths)
        discounted = payoffs * dfs
        # Optional control variate: we discounte terminal spot
        if self.control_variate:
            S = paths.get("S", None)
            df_path = paths.get("df", None)

            if S is not None and df_path is not None:
                ST = S[:, -1]
                dfT = df_path[:, -1]

                # control variable Y = DF * S_T, with known mean E[Y] = S0 (assuming q=0)
                Y = dfT * ST
                muY = float(model.s0)

                X = discounted

                varY = float(np.var(Y, ddof=1))
                if varY > 1e-16:
                    covXY = float(np.cov(X, Y, ddof=1)[0, 1])
                    beta = covXY / varY
                    discounted = X - beta * (Y - muY)

        price = float(np.mean(discounted))
        se = float(np.std(discounted, ddof=1) / np.sqrt(discounted.shape[0]))
        return price, se
