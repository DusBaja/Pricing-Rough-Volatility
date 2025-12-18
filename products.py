from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from observation import ObservationFrequency, get_observation_indices


class BaseProduct(ABC):
    """
    Abstract base class for all products.
    Holds a link to a model and defines
    the interface to compute payoff from simulated paths.
    """

    model: Optional["Model"] = None  # type: ignore[name-defined]

    def set_model(self, model: "Model") -> "BaseProduct":  # type: ignore[name-defined]
        self.model = model
        return self

    @property
    @abstractmethod
    def maturity_years(self) -> float:
        ...

    @abstractmethod
    def payoff_from_paths(
        self,
        paths: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Should return:
        - payoffs: shape (n_paths,)
        - discount_factors: shape (n_paths,)
        """
        ...


@dataclass
class AutocallAthenaProduct(BaseProduct):
    """
    Athena autocall product.

    Coupons are effectively paid at the termination date if the call_barrier is reached.
    The 'memory effect' determines whether the coupon compensate accrue linearly all the coupons missed, ie
    - with_memory = True:
        coupon_factor = coupon_per_year * t_years
    - with_memory = False:
        coupon_factor = coupon_per_year
    """
    #The nominal amount is just to take it into 100% at the end
    nominal: float
    strike: float
    coupon_per_year: float
    maturity_years_: float
    obs_freq: ObservationFrequency
    call_barrier: float
    protection_barrier: float
    steps_per_year: int
    with_memory: bool = False  

    model: Optional["Model"] = None  # type: ignore[name-defined]

    @property
    def maturity_years(self) -> float:
        return self.maturity_years_

    def payoff_from_paths(
        self,
        paths: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        S_paths = paths["S"]   # shape (n_paths, n_steps+1)
        df_paths = paths["df"] # shape (n_paths, n_steps+1)

        n_paths, n_steps_plus = S_paths.shape
        n_steps = n_steps_plus - 1

        # Observation including the final one
        obs_indices, _ = get_observation_indices(maturity_years=self.maturity_years,steps_per_year=self.steps_per_year,freq=self.obs_freq,)

        dt = self.maturity_years / n_steps

        payoffs = np.zeros(n_paths)
        discounts = np.zeros(n_paths)

        call_level = self.call_barrier * self.strike
        prot_level = self.protection_barrier * self.strike

        for i in range(n_paths): #number of path simulation
            path_S = S_paths[i]
            path_df = df_paths[i]
            autocalled = False

            # All observation dates except final maturity
            for idx in obs_indices[:-1]:
                ST = path_S[idx]
                DF_t = path_df[idx]
                t_years = idx * dt

                if ST >= call_level:
                    # Memory effect: coupon accrues with time
                    if self.with_memory:
                        coupon_factor = self.coupon_per_year * t_years
                    else:
                        # One "annual" coupon, regardless of when we call
                        coupon_factor = self.coupon_per_year

                    payoff = self.nominal * (1.0 + coupon_factor)
                    payoffs[i] = payoff
                    discounts[i] = DF_t
                    autocalled = True
                    break

            # If never autocalled, check maturity
            if not autocalled:
                idx_T = obs_indices[-1]
                ST = path_S[idx_T]
                DF_T = path_df[idx_T]
                t_years = idx_T * dt  # this should be close to maturity_years

                if ST >= prot_level:
                    # Above protection: receive nominal + coupons
                    if self.with_memory:
                        coupon_factor = self.coupon_per_year * t_years
                    else:
                        coupon_factor = self.coupon_per_year
                    payoff = self.nominal * (1.0 + coupon_factor)
                else:
                    # Below protection: capital at risk
                    payoff = self.nominal * (ST / self.strike)

                payoffs[i] = payoff
                discounts[i] = DF_T

        return payoffs, discounts

@dataclass
class AutocallPhoenixProduct(BaseProduct):
    """
    Phoenix autocall product with periodic coupons and optional memory effect.

    Parameters
    ----------
    nominal : float
        Notional amount.
    strike : float
        Initial spot / strike used for barriers and capital at risk.
    coupon_per_year : float
        Annualized coupon rate (e.g. 0.10 for 10% p.a.).
    maturity_years_ : float
        Maturity in years.
    obs_freq : ObservationFrequency
        Observation frequency (e.g. MONTHLY, QUARTERLY).
    call_barrier : float
        Call barrier as a fraction of strike (e.g. 1.0 for 100% of strike).
    coupon_barrier : float
        Coupon barrier as a fraction of strike.
    protection_barrier : float
        Protection barrier at maturity as a fraction of strike.
    steps_per_year : int
        Time discretization used in the model simulation.
    with_memory : bool, default False
        If True: missed coupons accumulate and are paid when barrier is hit.
        If False: each observation pays at most one coupon, no accumulation.
    """

    nominal: float
    strike: float
    coupon_per_year: float
    maturity_years_: float
    obs_freq: ObservationFrequency
    call_barrier: float
    coupon_barrier: float
    protection_barrier: float
    steps_per_year: int
    with_memory: bool = False

    @property
    def maturity_years(self) -> float:
        return self.maturity_years_

    def payoff_from_paths(
        self,
        paths: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Product has no model attached.")

        S_paths = paths.get("S")
        df_paths = paths.get("df")
        if S_paths is None or df_paths is None:
            raise ValueError("Paths must contain 'S' and 'df' for Phoenix product.")

        n_paths, n_steps_plus_1 = S_paths.shape
        n_steps = n_steps_plus_1 - 1
        if n_steps <= 0:
            raise ValueError("Phoenix: number of steps must be >= 1")

        
        obs_indices, _ = get_observation_indices(maturity_years=self.maturity_years,steps_per_year=self.steps_per_year,freq=self.obs_freq)
        obs_per_year = float(self.obs_freq.value)
        coupon_per_period = self.coupon_per_year / obs_per_year #as per annum

        call_level = self.call_barrier * self.strike
        coupon_level = self.coupon_barrier * self.strike
        prot_level = self.protection_barrier * self.strike

        # We return present value per path, so discount = 1 for pheonix 
        payoffs = np.zeros(n_paths)
        discounts = np.ones(n_paths)

        for i in range(n_paths):
            path_S = S_paths[i]
            path_df = df_paths[i]

            pv = 0.0
            missed_coupons = 0  # only used if with_memory = True
            autocalled = False

            for j, idx in enumerate(obs_indices):
                ST = path_S[idx]
                DF_t = path_df[idx]
                is_last = (j == len(obs_indices) - 1)

                
                if ST >= coupon_level:
                    if self.with_memory:
                        coupon_count = 1 + missed_coupons
                        missed_coupons = 0
                    else:
                        coupon_count = 1

                    coupon_amount = self.nominal * coupon_per_period * coupon_count
                    pv += coupon_amount * DF_t
                else:
                    if self.with_memory:
                        missed_coupons += 1

                
                if ST >= call_level and not is_last:
                    # At early call date we pay nominal.
                    # Coupon for this date (and memory) already added above if ST >= coupon_level.
                    pv += self.nominal * DF_t
                    autocalled = True
                    break

                
                if is_last and not autocalled:
                    # Coupon for final date (and memory) already handled above if the call_barrier is reached.
                    if ST >= prot_level:
                        redemption = self.nominal
                    else:
                        redemption = self.nominal * (ST / self.strike)

                    pv += redemption * DF_t
                    autocalled = True
                    break

            payoffs[i] = pv

        return payoffs, discounts