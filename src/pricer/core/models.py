from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np
import math


class Model:
    """
    Model class with the following available models:

    - GBM spot + flat rate (classic one: 1 factor sto spot)
    - Heston spot + flat rate (classic Heston: 2 factors sto ie vol and spot)
    - Heston spot + Hull–White rate (hybrid Heston: 3 factors sto ie vol, spot and rate)
    - Pure rate (flat or Hull–White: to be able to decompose rate effect)

    The spot dynamics are always exponential (GBM resolution):dS_t = S_t (r_t dt + sqrt(v_t) dW_S)
    where:
        - GBM: v_t = sigma^2 (constant).
        - Heston: v_t follows the Heston SDE.
        - Different rough models whose dynamic we indicate. 
    The rate dynamics are:
        - FLAT: r_t = r0
        - HULLWHITE: dr_t = a (b - r_t) dt + sigma_r dW_r
    """
    pricer: Optional["MonteCarloPricer"] = None   # type: ignore[name-defined]
    greeks: Optional["Greeks"] = None             # type: ignore[name-defined]

    def __init__(
        self,
        *,
        spot_process: Optional[str] = "GBM",      # "GBM", "HESTON" or None
        rate_process: Optional[str] = "FLAT",     # "FLAT", "HULLWHITE", or None
        vol_process: Optional[str] = "FLAT",      # "FLAT", "HESTON", "ROUGH_FBM","RFSV","RBERGOMI"  or None

        steps_per_year: int = 252,
        n_paths: int = 20_000,
        seed: Optional[int] = 42,

        s0: float = 100.0,
        sigma: float = 0.20,  # used when spot_process="GBM"
        xi0_times: Optional[np.ndarray] = None,
        xi0_values: Optional[np.ndarray] = None,
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
        rough_H: float = 0.10,        # Hurst exponent, 0 < H < 0., we fixed it based on Rosembaum's paper for better calibration (one param less)
        rough_nu: float = 0.30,       # vol-of-vol in log space
        rough_alpha: float = 5e-4,    # mean reversion speed for RFSV (per year)
        rough_m: Optional[float] = None,  # mean of X_t; defaults to log(sigma)
        rough_rho: float = -0.70,    # leverage corr between spot and vol driver (rough Bergomi)

        r0: float = 0.02,             # flat and initial short rate
        disc_times: Optional[np.ndarray] = None,
        disc_rates: Optional[np.ndarray] = None,
        
        # Hull–White
        a: float = 0.1,
        b: float = 0.02,
        sigma_r: float = 0.01,

        # cross correlations (for Heston+Hull–White hybrid)
        rho_sr: float = 0.3,
        rho_vr: float = 0.2):
    
        self.spot_process = spot_process.upper() if spot_process else None
        self.rate_process = rate_process.upper() if rate_process else None
        self.vol_process = vol_process.upper() if vol_process else None
        
        if self.vol_process == "ROUGH":
            self.vol_process = "ROUGH_FBM"
        
        self.steps_per_year = steps_per_year
        self.n_paths = n_paths
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.s0 = s0
        self.sigma = sigma
        self.xi0_times = None if xi0_times is None else np.asarray(xi0_times, dtype=float)
        self.xi0_values = None if xi0_values is None else np.asarray(xi0_values, dtype=float)

        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho_sv = rho_sv

        self.rough_H = rough_H
        self.rough_nu = rough_nu
        self.rough_alpha = rough_alpha
        # if rough_m is not provided, we center X_t around log(sigma)
        #self.rough_m = np.log(sigma) if rough_m is None else rough_m
        # rough_m is an offset in log-space.
        # The effective log level used by rough models is: log(sigma) + rough_m
        # If not provided, default to 0.0 so sigma remains the baseline.
        self.rough_m = 0.0 if rough_m is None else float(rough_m)
        self.rough_rho = float(rough_rho)

        self.r0 = r0
        self.disc_times = None if disc_times is None else np.asarray(disc_times, dtype=float)
        self.disc_rates = None if disc_rates is None else np.asarray(disc_rates, dtype=float)
        self.a = a
        self.b = b
        self.sigma_r = sigma_r

        # cross correlations
        self.rho_sr = rho_sr
        self.rho_vr = rho_vr
        self._chol_2 = None   # for (S, v)
        self._chol_3 = None   # for (S, v, r)

        self._build_correlations()

    def build_correlation(self) -> None:
        self._build_correlations()

    def _build_correlations(self) -> None:
        self._chol_2 = None
        self._chol_3 = None

        if self.spot_process == "HESTON" and self.rate_process in (None, "FLAT"):
            rho = self.rho_sv
            cov = np.array([[1.0, rho], [rho, 1.0]])
            self._chol_2 = np.linalg.cholesky(cov)

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

        if self.spot_process == "GBM" and self.rate_process == "HULLWHITE" and self.vol_process == "RBERGOMI":
          rho_sv = self.rough_rho   # spot vol corr
          rho_sr = self.rho_sr      # spot rate corr
          rho_vr = self.rho_vr      # rate vol corr

          cov = np.array(
              [
                  [1.0,    rho_sv, rho_sr],
                  [rho_sv, 1.0,    rho_vr],
                  [rho_sr, rho_vr, 1.0   ],
              ]
          )
          self._chol_3 = np.linalg.cholesky(cov)

    
    def simulate_paths(self,maturity_years: float,antithetic: bool = False) -> Dict[str, np.ndarray]:
        """
        Simulation interface for all model configurations.

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

            # rough Bergomi (Volterra) stochastic volatility
            if self.vol_process == "RBERGOMI":
                return self._simulate_gbm_rbergomi_flat(maturity_years, antithetic)

            raise ValueError(
                f"Unsupported vol_process={self.vol_process} "
                f"for spot_process=GBM, rate_process={self.rate_process}"
            )


        if self.spot_process == "HESTON" and self.rate_process in (None, "FLAT"):
            return self._simulate_heston_flat(maturity_years, antithetic)

        # Heston + Hull–White hybrid
        if self.spot_process == "HESTON" and self.rate_process == "HULLWHITE":
            return self._simulate_heston_hullwhite(maturity_years, antithetic)
        
        # GBM spot + Hull–White rate and Bergomi:
        if self.spot_process == "GBM" and self.rate_process == "HULLWHITE":
            #  GBM + HW with constant vol
            if self.vol_process in (None, "FLAT"):
                return self._simulate_gbm_hw(maturity_years, antithetic)

            #  GBM + rBergomi + HW
            if self.vol_process == "RBERGOMI":
                return self._simulate_gbm_rbergomi_hullwhite(maturity_years, antithetic)

            raise ValueError(
                f"Unsupported vol_process={self.vol_process} for spot_process=GBM, rate_process=HULLWHITE"
            )

        raise ValueError(
            f"Unsupported combination: spot_process={self.spot_process}, "
            f"rate_process={self.rate_process}"
        )
    
    def _df_at_times(self, times: np.ndarray) -> np.ndarray:
        """Discount factor DF(0,t) evaluated at times (years).
        If disc_times/disc_rates provided: interpolate zero rates (cont. comp) and compute DF=exp(-r(t)*t).
        Otherwise uses flat DF=exp(-r0*t).
        """
        if self.disc_times is None or self.disc_rates is None:
            return np.exp(-float(self.r0) * times)

        t = self.disc_times
        r = self.disc_rates
        if t.ndim != 1 or r.ndim != 1 or t.shape[0] != r.shape[0] or t.shape[0] < 2:
            raise ValueError("disc_times and disc_rates must be 1D arrays of same length >= 2")

        if not np.all(np.diff(t) >= 0):
            idx = np.argsort(t)
            t = t[idx]
            r = r[idx]

        r_t = np.interp(times, t, r)
        r_t[times <= t[0]] = r[0]
        r_t[times >= t[-1]] = r[-1]
        return np.exp(-r_t * times)
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
        times = np.linspace(0.0, maturity_years, n_steps + 1)
        df_curve = self._df_at_times(times)

        # Curve-consistent drift: stepwise forward rates implied by DF(0,t)
        # Ensures E[DF(0,T) * S_T] = S0 when a curve is provided.
        log_df = np.log(np.maximum(df_curve, 1e-300))
        fwd_rates = -(log_df[1:] - log_df[:-1]) / dt  # length n_steps
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
        df[:] = df_curve[None, :]

        S[:, 0] = self.s0

        for k in range(n_steps):
            dW = Z[:, k] * sqrt_dt
            r_step = float(fwd_rates[k])
            drift = (r_step - 0.5 * self.sigma**2) * dt
            diff = self.sigma * dW
            S[:, k + 1] = S[:, k] * np.exp(drift + diff)

        return {"S": S, "df": df}
    def _simulate_gbm_hw(self, maturity_years: float, antithetic: bool):
        """
        Simulate GBM spot under Q with Hull–White short rate for discounting:
            dS/S = r_t dt + sigma dW^S
            dr   = a(b - r) dt + sigma_r dW^r
        with constant corr d<W^S, W^r> = rho_sr dt.

        Returns whatever your pricer expects (typically paths and discount factors).
        """
        

        # --- numerics ---
        steps = int(self.steps_per_year * maturity_years)
        dt = maturity_years / steps
        sqdt = np.sqrt(dt)

        n = int(self.n_paths)
        s0 = float(self.s0)

        # --- parameters ---
        sigma = float(self.sigma)

        a = float(self.hw_a if hasattr(self, "hw_a") else self.a)
        b = float(self.hw_b if hasattr(self, "hw_b") else self.b)
        sigma_r = float(self.hw_sigma_r if hasattr(self, "hw_sigma_r") else self.sigma_r)

        r0 = float(self.r0)
        rho = float(getattr(self, "rho_sr", 0.0))

        # safety
        rho = max(-1.0, min(1.0, rho))
        rho_ortho = np.sqrt(max(0.0, 1.0 - rho * rho))

        # --- allocate ---
        S = np.empty((n, steps + 1), dtype=float)
        r = np.empty((n, steps + 1), dtype=float)
        D = np.empty((n, steps + 1), dtype=float)

        S[:, 0] = s0
        r[:, 0] = r0
        D[:, 0] = 1.0

        rng = np.random.default_rng(getattr(self, "seed", None))

        # generate base normals; if antithetic, pair them
        def normals(size):
            z = rng.standard_normal(size)
            if antithetic:
                return np.concatenate([z, -z], axis=0)[:size[0], :size[1]]
            return z

        # For antithetic with n paths, easiest is to generate for n//2 and mirror
        # But keep it simple/robust:
        Zs = rng.standard_normal((n, steps))
        Zp = rng.standard_normal((n, steps))
        if antithetic:
            half = n // 2
            Zs[:half] = rng.standard_normal((half, steps))
            Zs[half:2*half] = -Zs[:half]
            Zp[:half] = rng.standard_normal((half, steps))
            Zp[half:2*half] = -Zp[:half]
            if 2*half < n:
                # last odd path
                Zs[-1] = rng.standard_normal((steps,))
                Zp[-1] = rng.standard_normal((steps,))

        # correlated rate increment driver
        Zr = rho * Zs + rho_ortho * Zp

        # --- evolve ---
        for k in range(steps):
            rk = r[:, k]

            # short rate Euler
            r[:, k + 1] = rk + a * (b - rk) * dt + sigma_r * sqdt * Zr[:, k]

            # discount factor (left-point)
            D[:, k + 1] = D[:, k] * np.exp(-rk * dt)

            # spot exact log step with rk
            S[:, k + 1] = S[:, k] * np.exp((rk - 0.5 * sigma * sigma) * dt + sigma * sqdt * Zs[:, k])

        # ---- return in your project’s expected format ----
        # Many of your simulators return a dict-like bundle; adapt if needed:
        return {
            "S": S,
            "r": r,
            "df": D,
            "dt": dt,
            "t": np.linspace(0.0, maturity_years, steps + 1),
        }

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
        times = np.linspace(0.0, maturity_years, n_steps + 1)
        df_curve = self._df_at_times(times)

        log_df = np.log(np.maximum(df_curve, 1e-300))
        fwd_rates = -(log_df[1:] - log_df[:-1]) / dt  # length n_steps
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
        df[:] = df_curve[None, :]
        S[:, 0] = self.s0

        # Brownian shocks for the spot
        if antithetic:
            n_base = n_paths // 2
            Z_half = self.rng.standard_normal(size=(n_base, n_steps))
            Z_S = np.vstack([Z_half, -Z_half])
        else:
            Z_S = self.rng.standard_normal(size=(n_paths, n_steps))

        for k in range(n_steps):
            r_t = float(fwd_rates[k])
            sigma_k = sigma_t[:, k]
            dW_S = Z_S[:, k] * sqrt_dt

            drift = (r_t - 0.5 * sigma_k ** 2) * dt
            diff = sigma_k * dW_S

            S[:, k + 1] = S[:, k] * np.exp(drift + diff)

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
        times = np.linspace(0.0, maturity_years, n_steps + 1)
        df_curve = self._df_at_times(times)

        log_df = np.log(np.maximum(df_curve, 1e-300))
        fwd_rates = -(log_df[1:] - log_df[:-1]) / dt  # length n_steps
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
        df[:] = df_curve[None, :]
        S[:, 0] = self.s0

        if antithetic:
            n_base = n_paths // 2
            Z_half = self.rng.standard_normal(size=(n_base, n_steps))
            Z_S = np.vstack([Z_half, -Z_half])
        else:
            Z_S = self.rng.standard_normal(size=(n_paths, n_steps))

        for k in range(n_steps):
            r_t = float(fwd_rates[k])
            sigma_k = sigma_t[:, k]
            dW_S = Z_S[:, k] * sqrt_dt

            drift = (r_t - 0.5 * sigma_k ** 2) * dt
            diff = sigma_k * dW_S

            S[:, k + 1] = S[:, k] * np.exp(drift + diff)

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
        times = np.linspace(0.0, maturity_years, n_steps + 1)
        df_curve = self._df_at_times(times)

        log_df = np.log(np.maximum(df_curve, 1e-300))
        fwd_rates = -(log_df[1:] - log_df[:-1]) / dt  # length n_steps
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
        df[:] = df_curve[None, :]

        S[:, 0] = self.s0
        v[:, 0] = self.v0

        for k in range(n_steps):
            dW = Z[:, k, :] @ self._chol_2.T * sqrt_dt
            dW_S = dW[:, 0]
            dW_v = dW[:, 1]

            v_t = v[:, k]
            sqrt_v = np.sqrt(np.maximum(v_t, 0.0))
            dv = self.kappa * (self.theta - v_t) * dt + self.xi * sqrt_v * dW_v
            v_next = np.maximum(v_t + dv, 1e-12)

            r_t = float(fwd_rates[k])

            dlogS = (r_t - 0.5 * v_t) * dt + sqrt_v * dW_S
            S_next = S[:, k] * np.exp(dlogS)


            S[:, k + 1] = S_next
            v[:, k + 1] = v_next

        return {"S": S, "v": v, "df": df}


    # --------------------------------------------------------
    # GBM spot + rough Bergomi variance (flat rate)
    # --------------------------------------------------------
    def _xi0_at_times(self, times: np.ndarray) -> np.ndarray:
        """Forward variance curve xi0(t) evaluated at times (years).

        - If xi0_times/xi0_values are provided, uses linear interpolation with flat extrapolation.
        - Otherwise defaults to constant sigma^2.
        """
        if getattr(self, "xi0_times", None) is None or getattr(self, "xi0_values", None) is None:
            return np.full_like(times, float(self.sigma) ** 2, dtype=float)

        t = self.xi0_times
        x = self.xi0_values
        if t.ndim != 1 or x.ndim != 1 or t.shape[0] != x.shape[0] or t.shape[0] < 2:
            raise ValueError("xi0_times and xi0_values must be 1D arrays of same length >= 2")

        # Ensure sorted
        if not np.all(np.diff(t) >= 0):
            idx = np.argsort(t)
            t = t[idx]
            x = x[idx]

        # Linear interpolation + flat extrapolation
        out = np.interp(times, t, x)
        out[times <= t[0]] = x[0]
        out[times >= t[-1]] = x[-1]
        return out
    def _fwd_rate_at_times(self, times: np.ndarray) -> np.ndarray:
        """Instantaneous forward rate f(0,t) from DF(0,t).

        Computed by finite differences on -log DF. Uses small epsilon and is stable for simulation grid.
        """
        eps = 1e-5
        t1 = np.maximum(times - eps, 0.0)
        t2 = times + eps
        df1 = self._df_at_times(t1)
        df2 = self._df_at_times(t2)
        # f(0,t) ≈ ( -log DF(t2) + log DF(t1) ) / (t2 - t1)
        return (-np.log(df2) + np.log(df1)) / (t2 - t1)
    def _simulate_gbm_rbergomi_flat(
        self,
        maturity_years: float,
        antithetic: bool,
    ) -> Dict[str, np.ndarray]:
        """
        Rough Bergomi-style model (discrete-time Volterra approximation).

        We simulate a Volterra Gaussian process:
            Y_t = sqrt(2H) * ∫_0^t (t-s)^(H-1/2) dW^v_s

        and set the instantaneous variance to
            v_t = xi0(t) * exp( eta * Y_t - 0.5 * eta^2 * t^(2H) )

        The spot is simulated under risk-neutral dynamics
            dS_t = S_t * ( r dt + sqrt(v_t) dW^S_t )

        with corr(dW^S, dW^v) = rho.

        Notes:
        - This is a proper rough Bergomi volatility construction (Volterra rough driver + leverage).
        - The Volterra integral is approximated by a Riemann sum on the simulation grid (O(n_steps^2)).
          For calibration you will likely replace this with a faster hybrid / convolution scheme later.
        """
        n_steps = int(maturity_years * self.steps_per_year)
        if n_steps <= 0:
            raise ValueError("maturity_years * steps_per_year must be >= 1")

        H = float(self.rough_H)
        if not (0.0 < H < 0.5):
            raise ValueError("rough_H must be in (0, 0.5)")

        eta = float(self.rough_nu)
        rho = float(self.rough_rho)
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rough_rho must be in [-1, 1]")

        dt = maturity_years / n_steps
        times = np.linspace(0.0, maturity_years, n_steps + 1)  # includes 0

        n_paths = self.n_paths
        if antithetic and n_paths % 2 != 0:
            raise ValueError("For antithetic, n_paths must be even.")

        # --- Brownian shocks: build correlated (dW_S, dW_v)
        if antithetic:
            n_base = n_paths // 2
            Z1_half = self.rng.standard_normal(size=(n_base, n_steps))
            Z2_half = self.rng.standard_normal(size=(n_base, n_steps))
            Z1 = np.vstack([Z1_half, -Z1_half])
            Z2 = np.vstack([Z2_half, -Z2_half])
        else:
            Z1 = self.rng.standard_normal(size=(n_paths, n_steps))
            Z2 = self.rng.standard_normal(size=(n_paths, n_steps))

        dW_v = np.sqrt(dt) * Z2
        dW_S = np.sqrt(dt) * (rho * Z2 + np.sqrt(max(0.0, 1.0 - rho * rho)) * Z1)

        # --- Volterra process Y on the grid (FFT convolution, O(n_steps log n_steps))
        # We need Y_i = sum_{k=0}^{i-1} g_{i-k} * dW_v[k], with g_j = sqrt(2H) * (j*dt)^(H-1/2), j=1..n_steps.
        # This is a standard 1D convolution of dW_v with kernel g (lags), taking the first n_steps outputs.

        # Precompute kernel g (length n_steps): g[0] corresponds to lag 1 (dt)
        j = np.arange(1, n_steps + 1, dtype=float)
        g = np.sqrt(2.0 * H) * (j * dt) ** (H - 0.5)  # (n_steps,)

        # FFT-based convolution (batch over paths)
        # Pad to length L >= n_steps + n_steps - 1
        L = 1 << int(math.ceil(math.log2(2 * n_steps - 1)))
        # rfft along time axis
        G = np.fft.rfft(np.pad(g, (0, L - n_steps)))
        DV = np.fft.rfft(np.pad(dW_v, ((0, 0), (0, L - n_steps))), axis=1)
        conv = np.fft.irfft(DV * G, n=L, axis=1)[:, :n_steps]  # (n_paths, n_steps)

        Y = np.zeros((n_paths, n_steps + 1), dtype=float)
        Y[:, 1:] = conv

        # --- Forward variance curve xi0(t)
        xi0 = self._xi0_at_times(times)  # variance

        t_pow = times ** (2.0 * H)
        expo = eta * Y - 0.5 * (eta ** 2) * t_pow
        expo = np.clip(expo, -50.0, 50.0)   # prevents overflow/underflow by putting a ball ie stopping time 
        v = xi0[None, :] * np.exp(expo)  # (n_paths, n_steps+1)

        # --- Spot and discount factors
        S = np.empty((n_paths, n_steps + 1), dtype=float)
        df = np.empty((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = self.s0
        
        df_curve = self._df_at_times(times)
        log_df = np.log(np.maximum(df_curve, 1e-300))
        fwd_rates = -(log_df[1:] - log_df[:-1]) / dt  # length n_steps
        df[:] = df_curve[None, :]
        # Drift uses curve-implied stepwise forwards for consistency with df_curve
        for k in range(n_steps):
            r_k = float(fwd_rates[k])  # curve-consistent drift
            vol = np.sqrt(np.maximum(v[:, k], 0.0))
            #S[:, k + 1] = S[:, k] * np.exp((r_k - 0.5 * vol * vol) * dt + vol * dW_S[:, k])
            if not np.isfinite(dt):
                raise ValueError(f"dt is not finite: {dt}")

            if not np.all(np.isfinite(vol)):
                bad = np.where(~np.isfinite(vol))[0][:10]
                print("k=", k, "bad vol idx:", bad)
                print("v[bad,k]=", v[bad, k])
                raise ValueError("vol has NaN/Inf")

            if not np.all(np.isfinite(dW_S[:, k])):
                bad = np.where(~np.isfinite(dW_S[:, k]))[0][:10]
                print("k=", k, "bad dW_S idx:", bad)
                raise ValueError("dW_S has NaN/Inf")

            incr = (r_k - 0.5 * vol * vol) * dt + vol * dW_S[:, k]

            if not np.all(np.isfinite(incr)):
                bad = np.where(~np.isfinite(incr))[0][:10]
                print("k=", k, "bad incr idx:", bad)
                print("r_k:", r_k, "dt:", dt)
                print("vol[bad]:", vol[bad])
                print("dW_S[bad,k]:", dW_S[bad, k])
                raise ValueError("incr has NaN/Inf")

            S[:, k + 1] = S[:, k] * np.exp(incr)

            if not np.all(np.isfinite(S[:, k + 1])):
                bad = np.where(~np.isfinite(S[:, k + 1]))[0][:10]
                print("k=", k, "bad S idx:", bad)
                print("S_prev:", S[bad, k])
                print("incr:", incr[bad])
                raise ValueError("S became NaN/Inf")
        return {"S": S, "v": v, "df": df}

    def _simulate_gbm_rbergomi_hullwhite(self,maturity_years: float, antithetic: bool) -> Dict[str, np.ndarray]:
        """
        rBergomi volatility + Hull–White short rate hybrid.

        - Vol driver: Volterra Gaussian process Y from dW_v
        - Variance:   v_t = xi0(t) * exp( eta * Y_t - 0.5 * eta^2 * t^(2H) )
        - Rate:       dr_t = a (b - r_t) dt + sigma_r dW_r
        - Discount:   df_{k+1} = df_k * exp(-r_k dt)   (pathwise)
        - Spot:       dlogS = (r_t - 0.5 v_t) dt + sqrt(v_t) dW_S
        """
        if self._chol_3 is None:
            # build in case params changed
            self._build_correlations()
        if self._chol_3 is None:
            raise RuntimeError("3D correlation matrix not built for rBergomi + HullWhite.")

        n_steps = int(maturity_years * self.steps_per_year)
        if n_steps <= 0:
            raise ValueError("maturity_years * steps_per_year must be >= 1")

        H = float(self.rough_H)
        if not (0.0 < H < 0.5):
            raise ValueError("rough_H must be in (0, 0.5)")

        eta = float(self.rough_nu)
        rho = float(self.rough_rho)
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rough_rho must be in [-1, 1]")

        dt = maturity_years / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity_years, n_steps + 1)

        n_paths = self.n_paths
        if antithetic and n_paths % 2 != 0:
            raise ValueError("For antithetic, n_paths must be even.")

        # --- Correlated Gaussian shocks (S, v, r)
        if antithetic:
            half = n_paths // 2
            Z_half = self.rng.standard_normal(size=(half, n_steps, 3))
            Z = np.vstack([Z_half, -Z_half])
        else:
            Z = self.rng.standard_normal(size=(n_paths, n_steps, 3))

        dW = (Z @ self._chol_3.T) * sqrt_dt  # (n_paths, n_steps, 3)
        dW_S = dW[:, :, 0]                  # spot driver
        dW_v = dW[:, :, 1]                  # vol driver
        dW_r = dW[:, :, 2]                  # rate driver

        # --- Build Volterra process Y via FFT convolution (same logic as flat version)
        # Kernel g_j = sqrt(2H) * (j*dt)^(H-1/2), j=1..n_steps
        j = np.arange(1, n_steps + 1, dtype=float)
        g = np.sqrt(2.0 * H) * (j * dt) ** (H - 0.5)  # (n_steps,)

        L = 1 << int(math.ceil(math.log2(2 * n_steps - 1)))
        G = np.fft.rfft(np.pad(g, (0, L - n_steps)))

        DV = np.fft.rfft(np.pad(dW_v, ((0, 0), (0, L - n_steps))), axis=1)
        conv = np.fft.irfft(DV * G, n=L, axis=1)[:, :n_steps]  # (n_paths, n_steps)

        Y = np.zeros((n_paths, n_steps + 1), dtype=float)
        Y[:, 1:] = conv

        # --- Forward variance curve xi0(t)
        xi0 = self._xi0_at_times(times)  # (n_steps+1,)

        t_pow = times ** (2.0 * H)
        expo = eta * Y - 0.5 * (eta ** 2) * t_pow
        expo = np.clip(expo, -50.0, 50.0)
        v = xi0[None, :] * np.exp(expo)  # (n_paths, n_steps+1)

        # --- Hull–White rate + pathwise DF
        r = np.empty((n_paths, n_steps + 1), dtype=float)
        df = np.empty((n_paths, n_steps + 1), dtype=float)
        r[:, 0] = float(self.r0)
        df[:, 0] = 1.0

        a = float(self.a)
        b = float(self.b)
        sigma_r = float(self.sigma_r)

        for k in range(n_steps):
            r_t = r[:, k]
            dr = a * (b - r_t) * dt + sigma_r * dW_r[:, k]
            r[:, k + 1] = r_t + dr

            # discount over [t_k, t_{k+1}] using r_t
            df[:, k + 1] = df[:, k] * np.exp(-r_t * dt)

        # --- Spot
        S = np.empty((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = float(self.s0)

        for k in range(n_steps):
            r_t = r[:, k]
            v_t = np.maximum(v[:, k], 0.0)
            vol = np.sqrt(v_t)

            dlogS = (r_t - 0.5 * v_t) * dt + vol * dW_S[:, k]
            S[:, k + 1] = S[:, k] * np.exp(dlogS)

        return {"S": S, "v": v, "r": r, "df": df}

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
        """
        has_curve = (self.disc_times is not None) and (self.disc_rates is not None)
        rate_key = "disc_rates_parallel" if has_curve else "r0"

        
        if self.spot_process is None and self.rate_process == "FLAT":
            return [rate_key]

        if self.spot_process is None and self.rate_process == "HULLWHITE":
            return ["r0", "a", "b", "sigma_r"]

        # GBM spot + flat rate
        if self.spot_process == "GBM" and self.rate_process in (None, "FLAT"):
            if self.vol_process in (None, "FLAT"):
                return ["s0", "sigma", rate_key]

            if self.vol_process == "ROUGH_FBM":
                return ["s0", rate_key, "rough_H", "rough_nu"]

            if self.vol_process == "RFSV":
                return ["s0", rate_key, "rough_H", "rough_nu", "rough_alpha"]
            
            if self.vol_process == "RBERGOMI":
                return ["s0", "rough_nu", "rough_rho", "xi0_level", rate_key]
            # fallback
            return ["s0", "sigma", rate_key]

        # Heston spot + flat rate
        if self.spot_process == "HESTON" and self.rate_process in (None, "FLAT"):
            return ["s0", "v0", rate_key, "kappa", "theta", "xi", "rho_sv"]

        # Heston + Hull–White hybrid
        if self.spot_process == "HESTON" and self.rate_process == "HULLWHITE":
            return ["s0", "v0", "r0", "kappa", "theta", "xi", "rho_sv","a", "b", "sigma_r", "rho_sr", "rho_vr"]
        
        # Hybrid rough bergomi and HW on rates side
        if self.spot_process == "GBM" and self.rate_process == "HULLWHITE" and self.vol_process == "RBERGOMI":
            return ["s0", "rough_nu", "rough_rho", "xi0_level", "a", "b", "sigma_r", "rho_sr", "rho_vr"]


        return []


