# vol_models.py
import numpy as np

class GBMModel:
    """
    Classic Black–Scholes / GBM with flat rate and vol.
    dS_t = S_t (r dt + sigma dW_t)
    """
    def __init__(self, s0: float, r0: float, sigma: float,
                 steps_per_year: int = 252, n_paths: int = 50_000, seed: int | None = None):
        self.s0 = s0
        self.r0 = r0
        self.sigma = sigma
        self.steps_per_year = steps_per_year
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, maturity_years: float, antithetic: bool = False):
        n_steps = int(maturity_years * self.steps_per_year)
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

        r = self.r0
        sigma = self.sigma

        for k in range(n_steps):
            dW = Z[:, k] * sqrt_dt
            S_t = S[:, k]

            dlogS = (r - 0.5 * sigma ** 2) * dt + sigma * dW
            S[:, k + 1] = S_t * np.exp(dlogS)

            df[:, k + 1] = df[:, k] * np.exp(-r * dt)

        return S, df


class HestonModel:
    """
    Heston stochastic volatility with flat short rate r0.

    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t
    dS_t = S_t (r dt + sqrt(v_t) dW^S_t)
    corr(dW^S, dW^v) = rho_sv
    """

    def __init__(self,
                 s0: float,
                 v0: float,
                 r0: float,
                 kappa: float,
                 theta: float,
                 xi: float,
                 rho_sv: float,
                 steps_per_year: int = 252,
                 n_paths: int = 50_000,
                 seed: int | None = None):
        self.s0 = s0
        self.v0 = v0
        self.r0 = r0

        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho_sv = rho_sv

        self.steps_per_year = steps_per_year
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

        self._build_cholesky()

    def _build_cholesky(self):
        rho = self.rho_sv
        corr = np.array([[1.0, rho],
                         [rho, 1.0]])
        eps = 1e-10
        corr = corr + eps * np.eye(2)
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

        r = self.r0

        for k in range(n_steps):
            dW = np.einsum("ij,kj->ki", self.chol, Z[:, k, :]) * sqrt_dt
            dW_S = dW[:, 0]
            dW_v = dW[:, 1]

            v_t = np.maximum(v[:, k], 0.0)
            S_t = S[:, k]

            dv = self.kappa * (self.theta - v_t) * dt + self.xi * np.sqrt(v_t) * dW_v
            v_next = np.maximum(v_t + dv, 1e-12)

            dlogS = (r - 0.5 * v_t) * dt + np.sqrt(v_t) * dW_S
            S_next = S_t * np.exp(dlogS)

            df_next = df[:, k] * np.exp(-r * dt)

            S[:, k + 1] = S_next
            v[:, k + 1] = v_next
            df[:, k + 1] = df_next

        return S, df
