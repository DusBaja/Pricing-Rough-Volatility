# pricing.py
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any

import numpy as np

from models import Model
from products import BaseProduct


class MonteCarloPricer:
    """
    Generic Monte Carlo pricer.

    It does *not* know the details of the model or product.
    It only assumes:
        - model.simulate_paths(maturity_years, antithetic)
        - product.payoff_from_paths(paths)

    Optional: supports common random numbers via rng_state.
    """

    def __init__(self, antithetic: bool = False):
        self.antithetic = antithetic

    def price(
        self,
        product: BaseProduct,
        model: Optional[Model] = None,
        rng_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """
        Returns (price, standard_error).

        If `model` is None, it will try to use `product.model`.
        If `rng_state` is provided, resets model RNG to that state
        (common random numbers).
        """
        # Resolve model if not explicitly passed
        if model is None:
            if product.model is None:
                raise RuntimeError("No model provided to pricer, and product.model is None.")
            model = product.model

        # Common Random Numbers: reset RNG state if provided
        if rng_state is not None:
            model.rng.bit_generator.state = rng_state

        paths = model.simulate_paths(
            maturity_years=product.maturity_years,
            antithetic=self.antithetic,
        )

        payoffs, dfs = product.payoff_from_paths(paths)
        discounted = payoffs * dfs

        price = float(np.mean(discounted))
        se = float(np.std(discounted, ddof=1) / np.sqrt(discounted.shape[0]))
        return price, se
