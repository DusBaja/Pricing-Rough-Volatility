# greeks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Optional, Any

from models import Model
from products import BaseProduct
from pricing import MonteCarloPricer
import copy

@dataclass
class Greeks:
    """
    Greeks engine attached to a given model + pricer.

    Provides:
      - generic parameter sensitivities (dPrice/dParam),
      - classic Greeks (Delta, Gamma, Rho, Vega),
      - second-order / cross Greeks (Volga, Vanna).
    """

    model: Model
    pricer: MonteCarloPricer

    # --------------------------------------------------------
    # internal helper: revalue with optional common RNG state
    # --------------------------------------------------------

    def _revalue(self, product: BaseProduct, rng_state: Optional[Dict[str, Any]] = None) -> float:
        if rng_state is not None:
            # Reset to the same RNG state -> common random numbers
            self.model.rng.bit_generator.state = rng_state
        price, _ = self.pricer.price(product, self.model)
        return price


    def _get_rng_state(self):
        return copy.deepcopy(self.model.rng.bit_generator.state)

    # --------------------------------------------------------
    # parameter sensitivities
    # --------------------------------------------------------

    def parameter(
        self,
        product: BaseProduct,
        param_name: str,
        rel_bump: float = 0.01,
        central: bool = False,
    ) -> float:
        """
        dPrice / d(param_name) for any numeric attribute of the model.
        Uses common random numbers so finite differences are stable.
        """
        base_val = getattr(self.model, param_name)
        if base_val == 0:
            raise ValueError(f"Cannot apply relative bump on zero parameter '{param_name}'")

        bump = base_val * rel_bump

        # Capture RNG state ONCE for the whole greek computation
        base_state = self._get_rng_state()

        # Base price (with fixed RNG state)
        setattr(self.model, param_name, base_val)
        p0 = self._revalue(product, rng_state=base_state)

        # Up bump
        setattr(self.model, param_name, base_val + bump)
        p_up = self._revalue(product, rng_state=base_state)

        if central:
            # Down bump
            setattr(self.model, param_name, base_val - bump)
            p_dn = self._revalue(product, rng_state=base_state)
            sens = (p_up - p_dn) / (2.0 * bump)
        else:
            sens = (p_up - p0) / bump

        # Restore
        setattr(self.model, param_name, base_val)
        return sens

    def all_parameters(
        self,
        product: BaseProduct,
        param_names: Iterable[str],
        rel_bump: float = 0.01,
        central: bool = True,
    ) -> Dict[str, float]:
        return {
            name: self.parameter(product, name, rel_bump=rel_bump, central=central)
            for name in param_names
        }

    # --------------------------------------------------------
    # classic Greeks
    # --------------------------------------------------------

    def delta(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        central: bool = True,
        spot_attr: str = "s0",
    ) -> float:
        """
        dPrice/dS0 using finite differencing on the `s0` attribute of the model.
        """
        return self.parameter(
            product,
            param_name=spot_attr,
            rel_bump=rel_bump,
            central=central,
        )

    def gamma(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        spot_attr: str = "s0",
    ) -> float:
        """
        Second derivative wrt S0: Gamma, central difference, with common random numbers.
        """
        base_val = getattr(self.model, spot_attr)
        if base_val == 0:
            raise ValueError(f"Cannot apply relative bump on zero parameter '{spot_attr}'")

        h = base_val * rel_bump
        base_state = self._get_rng_state()

        # S + h
        setattr(self.model, spot_attr, base_val + h)
        p_up = self._revalue(product, rng_state=base_state)

        # S - h
        setattr(self.model, spot_attr, base_val - h)
        p_dn = self._revalue(product, rng_state=base_state)

        # S
        setattr(self.model, spot_attr, base_val)
        p0 = self._revalue(product, rng_state=base_state)

        gamma = (p_up - 2.0 * p0 + p_dn) / (h * h)
        return gamma

    def rho(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        central: bool = True,
        rate_attr: str = "r0",
    ) -> float:
        """
        Sensitivity wrt a rate parameter (typically `r0` for flat rate).
        Your scaling returns 'per bp' (as you intended).
        """
        return self.parameter(
            product,
            param_name=rate_attr,
            rel_bump=rel_bump,
            central=central,
        ) * 1e-4  # per bps

    def vega(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        central: bool = True,
        vol_attr: str = "sigma",
    ) -> float:
        """
        Vol sensitivity; returns 'per 1 vol point' (1%).
        """
        return self.parameter(
            product,
            param_name=vol_attr,
            rel_bump=rel_bump,
            central=central,
        ) * 0.01  # per vol point (1%)

    def volga(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        vol_attr: str = "sigma",
    ) -> float:
        """
        Second derivative wrt vol parameter (volga), with common random numbers.
        """
        base_val = getattr(self.model, vol_attr)
        if base_val == 0:
            raise ValueError(f"Cannot bump zero-vol parameter '{vol_attr}'")

        h = base_val * rel_bump
        base_state = self._get_rng_state()

        # σ + h
        setattr(self.model, vol_attr, base_val + h)
        p_up = self._revalue(product, rng_state=base_state)

        # σ - h
        setattr(self.model, vol_attr, base_val - h)
        p_dn = self._revalue(product, rng_state=base_state)

        # σ
        setattr(self.model, vol_attr, base_val)
        p0 = self._revalue(product, rng_state=base_state)

        # your scaling: divide by 100^2 to express per (1 vol point)^2
        volga = (p_up - 2.0 * p0 + p_dn) / (h * h * 1e4)
        print("p_up, p0, p_dn:", p_up, p0, p_dn, "second diff:", (p_up - 2*p0 + p_dn))

        return volga

    def vanna(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        spot_attr: str = "s0",
        vol_attr: str = "sigma",
    ) -> float:
        """
        Cross-derivative d^2Price/(dS dVol), with common random numbers.
        """
        s0 = getattr(self.model, spot_attr)
        vol = getattr(self.model, vol_attr)

        if s0 == 0 or vol == 0:
            raise ValueError("Cannot bump zero parameters for vanna.")

        hS = s0 * rel_bump
        hV = vol * rel_bump
        base_state = self._get_rng_state()

        # (S+hS, V+hV)
        setattr(self.model, spot_attr, s0 + hS)
        setattr(self.model, vol_attr, vol + hV)
        p_pp = self._revalue(product, rng_state=base_state)

        # (S+hS, V-hV)
        setattr(self.model, spot_attr, s0 + hS)
        setattr(self.model, vol_attr, vol - hV)
        p_pm = self._revalue(product, rng_state=base_state)

        # (S-hS, V+hV)
        setattr(self.model, spot_attr, s0 - hS)
        setattr(self.model, vol_attr, vol + hV)
        p_mp = self._revalue(product, rng_state=base_state)

        # (S-hS, V-hV)
        setattr(self.model, spot_attr, s0 - hS)
        setattr(self.model, vol_attr, vol - hV)
        p_mm = self._revalue(product, rng_state=base_state)

        # restore
        setattr(self.model, spot_attr, s0)
        setattr(self.model, vol_attr, vol)

        # your scaling: divide by 100 to express per 1 vol point
        vanna = (p_pp - p_pm - p_mp + p_mm) / (4.0 * hS * hV * 100.0)
        return vanna
