# rate_models.py
import numpy as np

class FlatRateModel:
    """
    Flat short rate r0.
    Mainly used to keep a unified interface with HullWhiteModel.
    """
    def __init__(self, r0: float):
        self.r0 = r0

    def simulate_paths(self, maturity_years: float, steps_per_year: int, n_paths: int, seed: int | None = None):
        n_steps = int(maturity_years * steps_per_year)
        dt = maturity_years / n_steps

        r = np.full((n_paths, n_steps + 1), self.r0)
        df = np.empty((n_paths, n_steps + 1))
        df[:, 0] = 1.0

        for k in range(n_steps):
            df[:, k + 1] = df[:, k] * np.exp(-self.r0 * dt)

        return r, df


class HullWhiteModel:
    """
    Hull–White 1-factor:
      dr_t = a (b - r_t) dt + sigma_r dW^r_t
    """
    def __init__(self, r0: float, a: float, b: float, sigma_r: float):
        self.r0 = r0
        self.a = a
        self.b = b
        self.sigma_r = sigma_r

    def simulate_paths(self, maturity_years: float, steps_per_year: int, n_paths: int, seed: int | None = None):
        rng = np.random.default_rng(seed)

        n_steps = int(maturity_years * steps_per_year)
        dt = maturity_years / n_steps
        sqrt_dt = np.sqrt(dt)

        r = np.empty((n_paths, n_steps + 1))
        df = np.empty((n_paths, n_steps + 1))

        r[:, 0] = self.r0
        df[:, 0] = 1.0

        a = self.a
        b = self.b
        sigma_r = self.sigma_r

        for k in range(n_steps):
            dW = rng.standard_normal(size=n_paths) * sqrt_dt
            r_k = r[:, k]

            dr = a * (b - r_k) * dt + sigma_r * dW
            r[:, k + 1] = r_k + dr

            df[:, k + 1] = df[:, k] * np.exp(-r_k * dt)

        return r, df
