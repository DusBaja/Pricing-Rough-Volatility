# models.py
from __future__ import annotations

from typing import Optional, Dict, Tuple

import numpy as np


class Model:
    """
    Unified model class that can represent:

    - GBM spot + flat rate
    - Heston spot + flat rate
    - Heston spot + Hull–White rate (hybrid)
    - Pure rate (flat or Hull–White)

    The spot dynamics are always exponential (GBM-style):

        dS_t = S_t (r_t dt + sqrt(v_t) dW_S)

    where:
        - GBM: v_t = sigma^2 (constant).
        - Heston: v_t follows the Heston SDE.

    The rate dynamics are:
        - FLAT: r_t = r0
        - HULLWHITE: dr_t = a (b - r_t) dt + sigma_r dW_r
    """

    # These are created later (e.g. in main.py) to avoid circular imports:
    #   model.pricer = MonteCarloPricer(...)
    #   model.greeks = Greeks(model, model.pricer)
    pricer: Optional["MonteCarloPricer"] = None   # type: ignore[name-defined]
    greeks: Optional["Greeks"] = None             # type: ignore[name-defined]

    def __init__(
        self,
        *,
        # which sub-models are active
        spot_process: Optional[str] = "GBM",      # "GBM", "HESTON" or None
        rate_process: Optional[str] = "FLAT",     # "FLAT", "HULLWHITE", or None
        vol_process: Optional[str] = "FLAT",      # "FLAT", "HESTON", "ROUGH_FBM","RFSV"  or None

        # common simulation controls
        steps_per_year: int = 252,
        n_paths: int = 20_000,
        seed: Optional[int] = 42,

        # Spot GBM parameters
        s0: float = 100.0,
        sigma: float = 0.20,          # used when spot_process="GBM"

        # Heston parameters (used when spot_process="HESTON")
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.5,
        rho_sv: float = -0.6,

        # --- Rough volatility (RFSV / fBM) parameters ---
        # log sigma_t is either:
        #   ROUGH_FBM: log sigma_t = log(sigma) + nu * W^H_t - 0.5 * nu^2 t^(2H)
        #   RFSV:      dX_t = nu dW^H_t - alpha (X_t - m) dt,  sigma_t = exp(X_t)
        rough_H: float = 0.10,        # Hurst exponent, 0 < H < 0.5
        rough_nu: float = 0.30,       # vol-of-vol in log space
        rough_alpha: float = 5e-4,    # mean reversion speed for RFSV (per year)
        rough_m: Optional[float] = None,  # mean of X_t; defaults to log(sigma)


        # rate parameters
        r0: float = 0.02,             # flat or initial short rate

        # Hull–White
        a: float = 0.1,
        b: float = 0.02,
        sigma_r: float = 0.01,

        # cross correlations (for Heston+Hull–White hybrid)
        rho_sr: float = 0.3,
        rho_vr: float = 0.2,
    ):
        # process types
        self.spot_process = spot_process.upper() if spot_process else None
        self.rate_process = rate_process.upper() if rate_process else None
        self.vol_process = vol_process.upper() if vol_process else None
        # Backward-compatible alias
        if self.vol_process == "ROUGH":
            self.vol_process = "ROUGH_FBM"

        # simulation controls
        self.steps_per_year = steps_per_year
        self.n_paths = n_paths
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # GBM / spot params
        self.s0 = s0
        self.sigma = sigma

        # Heston params
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho_sv = rho_sv

        # Rough volatility params
        self.rough_H = rough_H
        self.rough_nu = rough_nu
        self.rough_alpha = rough_alpha
        # if not provided, center X_t around log(sigma)
        #self.rough_m = np.log(sigma) if rough_m is None else rough_m
        # rough_m is an offset in log-space.
        # Effective log level used by rough models is: log(sigma) + rough_m
        # If not provided, default to 0.0 so sigma remains the baseline.
        self.rough_m = 0.0 if rough_m is None else float(rough_m)

        # rate params
        self.r0 = r0
        self.a = a
        self.b = b
        self.sigma_r = sigma_r

        # cross correlations
        self.rho_sr = rho_sr
        self.rho_vr = rho_vr

        # internal Cholesky factors
        self._chol_2 = None   # for (S, v)
        self._chol_3 = None   # for (S, v, r)

        self._build_correlations()

    # --------------------------------------------------------
    # internal: build correlations
    # --------------------------------------------------------

    def _build_correlations(self) -> None:
        self._chol_2 = None
        self._chol_3 = None

        # 2D correlation for Heston with flat rate
        if self.spot_process == "HESTON" and self.rate_process in (None, "FLAT"):
            rho = self.rho_sv
            cov = np.array([[1.0, rho], [rho, 1.0]])
            self._chol_2 = np.linalg.cholesky(cov)

        # 3D correlation for Heston + Hull-White
        if self.spot_process == "HESTON" and self.rate_process == "HULLWHITE":
            rho_sv = self.rho_sv
            rho_sr = self.rho_sr
            rho_vr = self.rho_vr

            cov = np.array(
                [
                    [1.0,    rho_sv, rho_sr],
                    [rho_sv, 1.0,    rho_vr],
                    [rho_sr, rho_vr, 1.0   ],
                ]
            )
            self._chol_3 = np.linalg.cholesky(cov)

    # --------------------------------------------------------
    # main simulation entry point
    # --------------------------------------------------------

    def simulate_paths(
        self,
        maturity_years: float,
        antithetic: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Unified simulation interface for all configurations.

        Returns a dictionary like:
            {
                "S":  S_paths (optional),
                "v":  v_paths (optional),
                "r":  r_paths (optional),
                "df": df_paths
            }
        """
        if self.spot_process is None and self.rate_process is None:
            raise ValueError("At least one of spot_process or rate_process must be set.")

        # pure rate models
        if self.spot_process is None and self.rate_process == "FLAT":
            return self._simulate_flat_rate(maturity_years, antithetic)

        if self.spot_process is None and self.rate_process == "HULLWHITE":
            return self._simulate_hullwhite_rate(maturity_years, antithetic)

                # spot + flat rate
        if self.spot_process == "GBM" and self.rate_process in (None, "FLAT"):
            # constant Black–Scholes vol
            if self.vol_process in (None, "FLAT"):
                return self._simulate_gbm_flat(maturity_years, antithetic)

            # simplified rough model: local fBM log-vol (α = 0)
            if self.vol_process == "ROUGH_FBM":
                return self._simulate_gbm_rough_fbm_flat(maturity_years, antithetic)

            # full RFSV: fractional OU log-vol
            if self.vol_process == "RFSV":
                return self._simulate_gbm_rfsV_flat(maturity_years, antithetic)

            raise ValueError(
                f"Unsupported vol_process={self.vol_process} "
                f"for spot_process=GBM, rate_process={self.rate_process}"
            )

        if self.spot_process == "HESTON" and self.rate_process in (None, "FLAT"):
            return self._simulate_heston_flat(maturity_years, antithetic)

        # Heston + Hull–White hybrid
        if self.spot_process == "HESTON" and self.rate_process == "HULLWHITE":
            return self._simulate_heston_hullwhite(maturity_years, antithetic)

        raise ValueError(
            f"Unsupported combination: spot_process={self.spot_process}, "
            f"rate_process={self.rate_process}"
        )

    # --------------------------------------------------------
    # pure rate: flat
    # --------------------------------------------------------

    def _simulate_flat_rate(
        self,
        maturity_years: float,
        antithetic: bool,
    ) -> Dict[str, np.ndarray]:
        n_steps = int(maturity_years * self.steps_per_year)
        if n_steps <= 0:
            raise ValueError("maturity_years * steps_per_year must be >= 1")

        dt = maturity_years / n_steps

        n_paths = self.n_paths
        if antithetic and n_paths % 2 != 0:
            raise ValueError("For antithetic, n_paths must be even.")

        r = np.full((n_paths, n_steps + 1), self.r0)
        df = np.empty((n_paths, n_steps + 1))
        df[:, 0] = 1.0

        for k in range(n_steps):
            df[:, k + 1] = df[:, k] * np.exp(-self.r0 * dt)

        return {"r": r, "df": df}

    # --------------------------------------------------------
    # pure rate: Hull–White
    # --------------------------------------------------------

    def _simulate_hullwhite_rate(
        self,
        maturity_years: float,
        antithetic: bool,
    ) -> Dict[str, np.ndarray]:
        n_steps = int(maturity_years * self.steps_per_year)
        if n_steps <= 0:
            raise ValueError("maturity_years * steps_per_year must be >= 1")

        dt = maturity_years / n_steps
        sqrt_dt = np.sqrt(dt)

        n_paths = self.n_paths
        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("For antithetic, n_paths must be even.")
            half = n_paths // 2
            Z_half = self.rng.standard_normal(size=(half, n_steps))
            Z = np.vstack([Z_half, -Z_half])
        else:
            Z = self.rng.standard_normal(size=(n_paths, n_steps))

        r = np.empty((n_paths, n_steps + 1))
        df = np.empty((n_paths, n_steps + 1))

        r[:, 0] = self.r0
        df[:, 0] = 1.0

        for k in range(n_steps):
            r_t = r[:, k]
            dW_r = Z[:, k] * sqrt_dt
            dr = self.a * (self.b - r_t) * dt + self.sigma_r * dW_r
            r_next = r_t + dr
            r[:, k + 1] = r_next

            df[:, k + 1] = df[:, k] * np.exp(-r_t * dt)

        return {"r": r, "df": df}

    def _generate_fbm_paths(
        self,
        n_steps: int,
        maturity_years: float,
        antithetic: bool,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Generate fBM paths W^H_t on [0, T] with T = maturity_years
        at n_steps time steps (plus t=0).

        Returns:
            BH_all:   (n_paths, n_steps+1) array with BH_all[:, 0] = 0
            times:    (n_steps+1,) time grid including 0
            dt:       time step size
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1")

        H = float(self.rough_H)
        if not (0.0 < H < 0.5):
            raise ValueError("rough_H must be in (0, 0.5)")

        n_paths = self.n_paths
        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("For antithetic, n_paths must be even.")
            n_base = n_paths // 2
        else:
            n_base = n_paths

        dt = maturity_years / n_steps
        times = np.linspace(dt, maturity_years, n_steps)  # t1...t_n
        t_col = times.reshape(-1, 1)

        # Covariance Γ_ij = 0.5 (t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})
        t_pow = t_col ** (2.0 * H)
        abs_diff = np.abs(t_col - t_col.T)
        Gamma = 0.5 * (t_pow + t_pow.T - abs_diff ** (2.0 * H))

        # numerical jitter
        eps = 1e-12
        Gamma[np.diag_indices_from(Gamma)] += eps

        L = np.linalg.cholesky(Gamma)  # (n_steps, n_steps)

        # simulate Gaussian vectors and map
        Z = self.rng.standard_normal(size=(n_steps, n_base))
        BH_tail = (L @ Z).T  # (n_base, n_steps)

        # prepend W^H_0 = 0
        BH_base = np.concatenate(
            [np.zeros((n_base, 1), dtype=float), BH_tail],
            axis=1,
        )  # (n_base, n_steps+1)

        if antithetic:
            BH_all = np.vstack([BH_base, -BH_base])  # (n_paths, n_steps+1)
        else:
            BH_all = BH_base

        times_full = np.concatenate([[0.0], times])  # (n_steps+1,)

        return BH_all, times_full, dt
    
    # --------------------------------------------------------
    # GBM spot + flat rate
    # --------------------------------------------------------

    def _simulate_gbm_flat(
        self,
        maturity_years: float,
        antithetic: bool,
    ) -> Dict[str, np.ndarray]:
        n_steps = int(maturity_years * self.steps_per_year)
        if n_steps <= 0:
            raise ValueError("maturity_years * steps_per_year must be >= 1")

        dt = maturity_years / n_steps
        sqrt_dt = np.sqrt(dt)

        n_paths = self.n_paths
        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("For antithetic, n_paths must be even.")
            half = n_paths // 2
            Z_half = self.rng.standard_normal(size=(half, n_steps))
            Z = np.vstack([Z_half, -Z_half])
        else:
            Z = self.rng.standard_normal(size=(n_paths, n_steps))

        S = np.empty((n_paths, n_steps + 1))
        df = np.empty((n_paths, n_steps + 1))

        S[:, 0] = self.s0
        df[:, 0] = 1.0

        for k in range(n_steps):
            dW = Z[:, k] * sqrt_dt
            drift = (self.r0 - 0.5 * self.sigma**2) * dt
            diff = self.sigma * dW
            S[:, k + 1] = S[:, k] * np.exp(drift + diff)
            df[:, k + 1] = df[:, k] * np.exp(-self.r0 * dt)

        return {"S": S, "df": df}
    def _simulate_gbm_rough_fbm_flat(
        self,
        maturity_years: float,
        antithetic: bool,
    ) -> Dict[str, np.ndarray]:
        """
        GBM spot with log-vol driven directly by fractional Brownian motion:

            log sigma_t = log(sigma) + nu * W^H_t - 0.5 * nu^2 t^{2H}

        This is the α = 0 local fBM approximation of the RFSV model.
        """
        n_steps = int(maturity_years * self.steps_per_year)
        BH_all, times_full, dt = self._generate_fbm_paths(
            n_steps, maturity_years, antithetic
        )
        sqrt_dt = np.sqrt(dt)

        H = float(self.rough_H)
        nu = float(self.rough_nu)

        # log sigma_t
        #log_sigma0 = np.log(self.sigma)
        log_sigma0 = np.log(self.sigma) + float(self.rough_m)
        t_pow_full = times_full ** (2.0 * H)
        drift_adj = 0.5 * (nu ** 2) * t_pow_full  # (n_steps+1,)
        log_sigma = log_sigma0 + nu * BH_all - drift_adj[None, :]
        sigma_t = np.exp(log_sigma)  # (n_paths, n_steps+1)

        n_paths = sigma_t.shape[0]

        # simulate GBM under flat rate r0 with path-dependent sigma
        S = np.empty((n_paths, n_steps + 1))
        df = np.empty((n_paths, n_steps + 1))
        S[:, 0] = self.s0
        df[:, 0] = 1.0

        # Brownian shocks for the spot
        if antithetic:
            n_base = n_paths // 2
            Z_half = self.rng.standard_normal(size=(n_base, n_steps))
            Z_S = np.vstack([Z_half, -Z_half])
        else:
            Z_S = self.rng.standard_normal(size=(n_paths, n_steps))

        r_t = self.r0

        for k in range(n_steps):
            sigma_k = sigma_t[:, k]
            dW_S = Z_S[:, k] * sqrt_dt

            drift = (r_t - 0.5 * sigma_k ** 2) * dt
            diff = sigma_k * dW_S

            S[:, k + 1] = S[:, k] * np.exp(drift + diff)
            df[:, k + 1] = df[:, k] * np.exp(-r_t * dt)

        # expose sigma paths so you can inspect/compare
        return {"S": S, "df": df, "sigma": sigma_t}
    # --------------------------------------------------------
    # GBM spot + flat rate with full RFSV (fOU) volatility
    # --------------------------------------------------------

    def _simulate_gbm_rfsV_flat(
        self,
        maturity_years: float,
        antithetic: bool,
    ) -> Dict[str, np.ndarray]:
        """
        GBM spot with log-vol following the fractional OU SDE:

            dX_t = nu dW^H_t - alpha (X_t - m) dt
            sigma_t = exp(X_t)

        Here W^H_t is the same fBM as in the simplified model.
        We discretize the SDE via Euler on the same grid.
        """
        n_steps = int(maturity_years * self.steps_per_year)
        BH_all, times_full, dt = self._generate_fbm_paths(
            n_steps, maturity_years, antithetic
        )
        sqrt_dt = np.sqrt(dt)

        H = float(self.rough_H)
        nu = float(self.rough_nu)
        alpha = float(self.rough_alpha)
        #m = float(self.rough_m)
        #m = float(self.rough_m) if self.rough_m is not None else float(np.log(self.sigma))
        # Effective long-run mean in log-space: log(sigma) + rough_m offset
        m = float(np.log(self.sigma) + float(self.rough_m))

        n_paths, n_grid = BH_all.shape  # n_grid = n_steps+1

        X = np.empty((n_paths, n_grid))
        X[:, 0] = m  # start at mean; with alpha small, burn-in is negligible

        for k in range(n_grid - 1):
            dBH = BH_all[:, k + 1] - BH_all[:, k]
            X[:, k + 1] = (
                X[:, k]
                + nu * dBH
                - alpha * (X[:, k] - m) * dt
            )

        sigma_t = np.exp(X)

        # simulate GBM with this stochastic volatility
        S = np.empty((n_paths, n_steps + 1))
        df = np.empty((n_paths, n_steps + 1))
        S[:, 0] = self.s0
        df[:, 0] = 1.0

        if antithetic:
            n_base = n_paths // 2
            Z_half = self.rng.standard_normal(size=(n_base, n_steps))
            Z_S = np.vstack([Z_half, -Z_half])
        else:
            Z_S = self.rng.standard_normal(size=(n_paths, n_steps))

        r_t = self.r0

        for k in range(n_steps):
            sigma_k = sigma_t[:, k]
            dW_S = Z_S[:, k] * sqrt_dt

            drift = (r_t - 0.5 * sigma_k ** 2) * dt
            diff = sigma_k * dW_S

            S[:, k + 1] = S[:, k] * np.exp(drift + diff)
            df[:, k + 1] = df[:, k] * np.exp(-r_t * dt)

        return {"S": S, "df": df, "sigma": sigma_t}
    # --------------------------------------------------------
    # Heston spot + flat rate
    # --------------------------------------------------------

    def _simulate_heston_flat(
        self,
        maturity_years: float,
        antithetic: bool,
    ) -> Dict[str, np.ndarray]:
        if self._chol_2 is None:
            raise RuntimeError("2D correlation matrix not built.")

        n_steps = int(maturity_years * self.steps_per_year)
        if n_steps <= 0:
            raise ValueError("maturity_years * steps_per_year must be >= 1")

        dt = maturity_years / n_steps
        sqrt_dt = np.sqrt(dt)

        n_paths = self.n_paths
        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("For antithetic, n_paths must be even.")
            half = n_paths // 2
            Z_half = self.rng.standard_normal(size=(half, n_steps, 2))
            Z = np.vstack([Z_half, -Z_half])
        else:
            Z = self.rng.standard_normal(size=(n_paths, n_steps, 2))

        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        df = np.empty((n_paths, n_steps + 1))

        S[:, 0] = self.s0
        v[:, 0] = self.v0
        df[:, 0] = 1.0

        for k in range(n_steps):
            dW = Z[:, k, :] @ self._chol_2.T * sqrt_dt
            dW_S = dW[:, 0]
            dW_v = dW[:, 1]

            v_t = v[:, k]
            sqrt_v = np.sqrt(np.maximum(v_t, 0.0))
            dv = self.kappa * (self.theta - v_t) * dt + self.xi * sqrt_v * dW_v
            v_next = np.maximum(v_t + dv, 1e-12)

            r_t = self.r0

            dlogS = (r_t - 0.5 * v_t) * dt + sqrt_v * dW_S
            S_next = S[:, k] * np.exp(dlogS)

            df_next = df[:, k] * np.exp(-r_t * dt)

            S[:, k + 1] = S_next
            v[:, k + 1] = v_next
            df[:, k + 1] = df_next

        return {"S": S, "v": v, "df": df}

    # --------------------------------------------------------
    # Heston + Hull–White hybrid
    # --------------------------------------------------------

    def _simulate_heston_hullwhite(
        self,
        maturity_years: float,
        antithetic: bool,
    ) -> Dict[str, np.ndarray]:
        if self._chol_3 is None:
            raise RuntimeError("3D correlation matrix not built.")

        n_steps = int(maturity_years * self.steps_per_year)
        if n_steps <= 0:
            raise ValueError("maturity_years * steps_per_year must be >= 1")

        dt = maturity_years / n_steps
        sqrt_dt = np.sqrt(dt)

        n_paths = self.n_paths
        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("For antithetic, n_paths must be even.")
            half = n_paths // 2
            Z_half = self.rng.standard_normal(size=(half, n_steps, 3))
            Z = np.vstack([Z_half, -Z_half])
        else:
            Z = self.rng.standard_normal(size=(n_paths, n_steps, 3))

        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        r = np.empty((n_paths, n_steps + 1))
        df = np.empty((n_paths, n_steps + 1))

        S[:, 0] = self.s0
        v[:, 0] = self.v0
        r[:, 0] = self.r0
        df[:, 0] = 1.0

        for k in range(n_steps):
            dW = Z[:, k, :] @ self._chol_3.T * sqrt_dt
            dW_S = dW[:, 0]
            dW_v = dW[:, 1]
            dW_r = dW[:, 2]

            S_t = S[:, k]
            v_t = v[:, k]
            r_t = r[:, k]

            sqrt_v = np.sqrt(np.maximum(v_t, 0.0))
            dv = self.kappa * (self.theta - v_t) * dt + self.xi * sqrt_v * dW_v
            v_next = np.maximum(v_t + dv, 1e-12)

            dr = self.a * (self.b - r_t) * dt + self.sigma_r * dW_r
            r_next = r_t + dr

            dlogS = (r_t - 0.5 * v_t) * dt + sqrt_v * dW_S
            S_next = S_t * np.exp(dlogS)

            df_next = df[:, k] * np.exp(-r_t * dt)

            S[:, k + 1] = S_next
            v[:, k + 1] = v_next
            r[:, k + 1] = r_next
            df[:, k + 1] = df_next

        return {"S": S, "v": v, "r": r, "df": df}
    
    def default_sensitivity_parameters(self) -> list[str]:
        """
        Returns a list of parameter names that are actually used
        by the current (spot_process, rate_process) configuration.

        This is just a convenience for Greeks / main.py.
        """
        params: list[str] = []

        # Rate-only models
        if self.spot_process is None and self.rate_process == "FLAT":
            return ["r0"]

        if self.spot_process is None and self.rate_process == "HULLWHITE":
            return ["r0", "a", "b", "sigma_r"]

        # GBM spot + flat rate
        if self.spot_process == "GBM" and self.rate_process in (None, "FLAT"):
            if self.vol_process in (None, "FLAT"):
                return ["s0", "sigma", "r0"]

            if self.vol_process == "ROUGH_FBM":
                # typically you might bump H and nu
                return ["s0", "sigma", "r0", "rough_H", "rough_nu"]

            if self.vol_process == "RFSV":
                return ["s0", "sigma", "r0", "rough_H", "rough_nu", "rough_alpha"]

            # Fallback:
            return ["s0", "sigma", "r0"]

        # Heston spot + flat rate
        if self.spot_process == "HESTON" and self.rate_process in (None, "FLAT"):
            return ["s0", "v0", "r0", "kappa", "theta", "xi", "rho_sv"]

        # Heston + Hull–White hybrid
        if self.spot_process == "HESTON" and self.rate_process == "HULLWHITE":
            return [
                "s0",
                "v0",
                "r0",
                "kappa",
                "theta",
                "xi",
                "rho_sv",
                "a",
                "b",
                "sigma_r",
                "rho_sr",
                "rho_vr",
            ]

        
        return params


