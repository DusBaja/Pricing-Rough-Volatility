from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Optional, Any, Tuple

import copy

from pricer.core.models import Model
from pricer.core.products import BaseProduct
from pricer.core.pricing import MonteCarloPricer


@dataclass
class Greeks:
    """Finite-difference Greeks using common random numbers (CRN).

    Pseudo-parameters supported by `parameter()`:
      - disc_rates_parallel: absolute +1bp parallel shift of discount curve zero rates
      - xi0_level: multiplicative scaling of xi0_values by (1 ± rel_bump)

    Conventions for reporting:
      - Delta: dP/dS0
      - Gamma: d^2P/dS0^2
      - Rho: curve DV01 per 1bp when a curve is attached, else flat-rate per 1bp
      - Vega(vol_param): first-order sensitivity to a chosen 'vol knob', reported per +1% move
        (for sigma this is per 1 vol point = 1%).
      - Volga(vol_param): second-order wrt the chosen vol knob, per (1%)^2
      - Vanna(vol_param): cross derivative d^2P/(dS d vol_knob), per 1% in the vol knob
    """

    model: Model
    pricer: MonteCarloPricer


    def _get_rng_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.model.rng.bit_generator.state)

    def _revalue(self, product: BaseProduct, rng_state: Optional[Dict[str, Any]] = None) -> float:
        if rng_state is not None:
            self.model.rng.bit_generator.state = rng_state
        price, _ = self.pricer.price(product, self.model)
        return float(price)


    def parameter(
        self,
        product: BaseProduct,
        param_name: str,
        rel_bump: float = 0.01,
        central: bool = False,
    ) -> float:
        """Finite-difference sensitivity dPrice/d(param) using CRN.

        - For scalar parameters: relative bump base*(1±rel_bump).
        - For 'disc_rates_parallel': absolute +1bp shift (bump=1e-4) of model.disc_rates.
        - For 'xi0_level': multiplicative bump of model.xi0_values by (1±rel_bump).

        Returns derivative per *one unit* of the parameter.
        """

        base_state = self._get_rng_state()

        #parallel discount-curve bump 
        if param_name == "disc_rates_parallel":
            if self.model.disc_times is None or self.model.disc_rates is None:
                raise ValueError("disc_rates_parallel requires model.disc_times and model.disc_rates")

            base_rates = self.model.disc_rates.copy()
            bump = 1e-4  # 1bp

            self.model.disc_rates = base_rates
            p0 = self._revalue(product, rng_state=base_state)

            self.model.disc_rates = base_rates + bump
            p_up = self._revalue(product, rng_state=base_state)

            if central:
                self.model.disc_rates = base_rates - bump
                p_dn = self._revalue(product, rng_state=base_state)
                sens = (p_up - p_dn) / (2.0 * bump)
            else:
                sens = (p_up - p0) / bump

            self.model.disc_rates = base_rates
            return float(sens)

        
        if param_name == "xi0_level":
            if self.model.xi0_values is None:
                raise ValueError("xi0_level requires model.xi0_values")

            base_xi = self.model.xi0_values.copy()
            bump = float(rel_bump)  # dimensionless

            self.model.xi0_values = base_xi
            p0 = self._revalue(product, rng_state=base_state)

            self.model.xi0_values = base_xi * (1.0 + bump)
            p_up = self._revalue(product, rng_state=base_state)

            if central:
                self.model.xi0_values = base_xi * (1.0 - bump)
                p_dn = self._revalue(product, rng_state=base_state)
                sens = (p_up - p_dn) / (2.0 * bump)
            else:
                sens = (p_up - p0) / bump

            self.model.xi0_values = base_xi
            return float(sens)

        base_val = getattr(self.model, param_name)
        if base_val == 0:
            raise ValueError(f"Cannot apply relative bump on zero parameter '{param_name}'")

        bump = base_val * rel_bump

        setattr(self.model, param_name, base_val)
        p0 = self._revalue(product, rng_state=base_state)

        setattr(self.model, param_name, base_val + bump)
        p_up = self._revalue(product, rng_state=base_state)

        if central:
            setattr(self.model, param_name, base_val - bump)
            p_dn = self._revalue(product, rng_state=base_state)
            sens = (p_up - p_dn) / (2.0 * bump)
        else:
            sens = (p_up - p0) / bump

        setattr(self.model, param_name, base_val)
        return float(sens)

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

    def delta(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        central: bool = True,
        spot_attr: str = "s0",
    ) -> float:
        return self.parameter(product, param_name=spot_attr, rel_bump=rel_bump, central=central)

    def gamma(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        spot_attr: str = "s0",
    ) -> float:
        s0 = getattr(self.model, spot_attr)
        if s0 == 0:
            raise ValueError(f"Cannot apply relative bump on zero parameter '{spot_attr}'")

        h = s0 * rel_bump
        base_state = self._get_rng_state()

        setattr(self.model, spot_attr, s0 + h)
        p_up = self._revalue(product, rng_state=base_state)

        setattr(self.model, spot_attr, s0 - h)
        p_dn = self._revalue(product, rng_state=base_state)

        setattr(self.model, spot_attr, s0)
        p0 = self._revalue(product, rng_state=base_state)

        return float((p_up - 2.0 * p0 + p_dn) / (h * h))

    def rho(
        self,
        product: BaseProduct,
        rel_bump: float = 0.01,
        central: bool = True,
        rate_attr: str = "r0",
    ) -> float:
        """Rate sensitivity per 1bp.

        If a discount curve is attached (disc_times/disc_rates), return curve DV01.
        Otherwise bump the flat rate attribute.
        """
        if self.model.disc_times is not None and self.model.disc_rates is not None:
            dP_drate = self.parameter(product, "disc_rates_parallel", rel_bump=rel_bump, central=central)
        else:
            dP_drate = self.parameter(product, param_name=rate_attr, rel_bump=rel_bump, central=central)
        return float(dP_drate) * 1e-4  # per bp


    def _vol_bump_size(self, vol_param: str, rel_bump: float) -> float:
        """Return the *step size* used for finite differences.

        - xi0_level uses dimensionless step h = rel_bump.
        - disc_rates_parallel is not a volatility knob.
        - scalar attributes use h = base * rel_bump.
        """
        if vol_param == "xi0_level":
            return float(rel_bump)
        if vol_param == "disc_rates_parallel":
            raise ValueError("disc_rates_parallel is not a vol knob")
        base = getattr(self.model, vol_param)
        if base == 0:
            raise ValueError(f"Cannot bump zero parameter '{vol_param}'")
        return float(base) * float(rel_bump)

    def _apply_vol_bump(self, vol_param: str, bump: float) -> Any:
        """Apply additive bump in the internal parameterization.

        For xi0_level: xi0 <- xi0*(1+bump) where bump is dimensionless.
        For scalar attrs: attr <- base + bump.

        Returns the base object to restore.
        """
        if vol_param == "xi0_level":
            if self.model.xi0_values is None:
                raise ValueError("xi0_level requires model.xi0_values")
            base = self.model.xi0_values.copy()
            self.model.xi0_values = base * (1.0 + bump)
            return base

        base = getattr(self.model, vol_param)
        setattr(self.model, vol_param, base + bump)
        return base

    def _restore_vol(self, vol_param: str, base: Any) -> None:
        if vol_param == "xi0_level":
            self.model.xi0_values = base
        else:
            setattr(self.model, vol_param, base)

    def vega(
        self,
        product: BaseProduct,
        vol_param: str = "sigma",
        rel_bump: float = 0.01,
        central: bool = True,
    ) -> float:
        """Sensitivity wrt chosen vol knob, reported per +1% move."""
        # parameter() returns derivative per 1 unit of the knobASSERT; convert to per 1%.
        dP_dparam = self.parameter(product, param_name=vol_param, rel_bump=rel_bump, central=central)
        return float(dP_dparam) * 0.01

    def volga(
        self,
        product: BaseProduct,
        vol_param: str = "sigma",
        rel_bump: float = 0.01,
    ) -> float:
        """Second derivative wrt chosen vol knob, reported per (1%)^2."""
        h = self._vol_bump_size(vol_param, rel_bump)
        base_state = self._get_rng_state()

        base = self._apply_vol_bump(vol_param, +h)
        p_up = self._revalue(product, rng_state=base_state)

        self._restore_vol(vol_param, base)
        base = self._apply_vol_bump(vol_param, -h)
        p_dn = self._revalue(product, rng_state=base_state)

        self._restore_vol(vol_param, base)
        p0 = self._revalue(product, rng_state=base_state)
        d2 = (p_up - 2.0 * p0 + p_dn) / (h * h)# d2P/dparam^2
        return float(d2) * 1e-4# per (1%)^2

    def vanna(
        self,
        product: BaseProduct,
        vol_param: str = "sigma",
        rel_bump: float = 0.01,
        spot_attr: str = "s0",
    ) -> float:
        """Cross derivative d^2P/(dS d vol_knob), reported per 1% in the vol knob."""
        s0 = getattr(self.model, spot_attr)
        if s0 == 0:
            raise ValueError("Cannot bump zero spot for vanna")

        hS = s0 * rel_bump
        hV = self._vol_bump_size(vol_param, rel_bump)
        base_state = self._get_rng_state()
        base_spot = s0
        base_vol = None
        if vol_param == "xi0_level":
            if self.model.xi0_values is None:
                raise ValueError("xi0_level requires model.xi0_values")
            base_vol = self.model.xi0_values.copy()
        else:
            base_vol = getattr(self.model, vol_param)

        # (S+hS, V+hV)
        setattr(self.model, spot_attr, base_spot + hS)
        self._apply_vol_bump(vol_param, +hV)
        p_pp = self._revalue(product, rng_state=base_state)
        self._restore_vol(vol_param, base_vol)

        # (S+hS, V-hV)
        setattr(self.model, spot_attr, base_spot + hS)
        self._apply_vol_bump(vol_param, -hV)
        p_pm = self._revalue(product, rng_state=base_state)
        self._restore_vol(vol_param, base_vol)

        # (S-hS, V+hV)
        setattr(self.model, spot_attr, base_spot - hS)
        self._apply_vol_bump(vol_param, +hV)
        p_mp = self._revalue(product, rng_state=base_state)
        self._restore_vol(vol_param, base_vol)

        # (S-hS, V-hV)
        setattr(self.model, spot_attr, base_spot - hS)
        self._apply_vol_bump(vol_param, -hV)
        p_mm = self._revalue(product, rng_state=base_state)
        self._restore_vol(vol_param, base_vol)

        setattr(self.model, spot_attr, base_spot)# restore spot
        d2 = (p_pp - p_pm - p_mp + p_mm) / (4.0 * hS * hV) # cross derivative d^2P/(dS dparam)

        return float(d2) * 0.01 # per 1% in the vol knob
