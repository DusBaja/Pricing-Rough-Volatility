# hybrid_models.py
import numpy as np

class HestonHullWhiteHybrid:
    """
    Heston stochastic volatility + Hull–White short rate with full correlation between
    [W_S, W_v, W_r].
    """

    def __init__(
        self,
        s0: float,
        v0: float,
        r0: float,
        # Heston params
        kappa: float,
        theta: float,
        xi: float,
        rho_sv: float,
        # Hull–White params
        a: float,
        b: float,
        sigma_r: float,
        # correlations with rates
        rho_sr: float = 0.0,
        rho_vr: float = 0.0,
        steps_per_year: int = 252,
        n_paths: int = 50_000,
        seed: int | None = None,
    ):
        self.s0 = s0
        self.v0 = v0
        self.r0 = r0

        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho_sv = rho_sv

        self.a = a
        self.b = b
        self.sigma_r = sigma_r
        self.rho_sr = rho_sr
        self.rho_vr = rho_vr

        self.steps_per_year = steps_per_year
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

        self._build_cholesky()

    def _build_cholesky(self):
        rho_sv = self.rho_sv
        rho_sr = self.rho_sr
        rho_vr = self.rho_vr

        corr = np.array([
            [1.0,     rho_sv, rho_sr],
            [rho_sv,  1.0,    rho_vr],
            [rho_sr,  rho_vr, 1.0   ]
        ])

        eps = 1e-10
        corr = corr + eps * np.eye(3)
        self.chol = np.linalg.cholesky(corr)

    def simulate_paths(self, maturity_years: float, antithetic: bool = False):
        n_steps = int(maturity_years * self.steps_per_year)
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
            # Z[:, k, :] has shape (n_paths, 3)
            # self.chol has shape (3, 3)
            # => dW has shape (n_paths, 3)
            dW = Z[:, k, :] @ self.chol.T * sqrt_dt

            dW_S = dW[:, 0]
            dW_v = dW[:, 1]
            dW_r = dW[:, 2]

            v_t = np.maximum(v[:, k], 0.0)
            r_t = r[:, k]
            S_t = S[:, k]

            dv = self.kappa * (self.theta - v_t) * dt + self.xi * np.sqrt(v_t) * dW_v
            v_next = np.maximum(v_t + dv, 1e-12)

            dr = self.a * (self.b - r_t) * dt + self.sigma_r * dW_r
            r_next = r_t + dr

            dlogS = (r_t - 0.5 * v_t) * dt + np.sqrt(v_t) * dW_S
            S_next = S_t * np.exp(dlogS)

            df_next = df[:, k] * np.exp(-r_t * dt)

            S[:, k + 1] = S_next
            v[:, k + 1] = v_next
            r[:, k + 1] = r_next
            df[:, k + 1] = df_next

        return S, v, r, df
