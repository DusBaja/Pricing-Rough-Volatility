# observation.py
from __future__ import annotations

from enum import Enum
from typing import Tuple, List

import numpy as np


class ObservationFrequency(Enum):
    """
    Frequency of product observations / call dates.
    """
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12

    def observations_per_year(self) -> int:
        return {
            ObservationFrequency.ANNUAL: 1,
            ObservationFrequency.SEMI_ANNUAL: 2,
            ObservationFrequency.QUARTERLY: 4,
            ObservationFrequency.MONTHLY: 12,
        }[self]


def get_observation_indices(
    maturity_years: float,
    steps_per_year: int,
    freq: ObservationFrequency,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - obs_indices: integer time indices (0..n_steps) of observation dates
      - obs_times:   the same dates in years (floats)

    Example: 5Y maturity, 252 steps/year, ANNUAL freq
      -> obs_indices ~ [252, 504, 756, 1008, 1260]
    """
    n_steps = int(round(maturity_years * steps_per_year))
    if n_steps <= 0:
        raise ValueError("maturity_years * steps_per_year must be >= 1")

    obs_per_year = freq.observations_per_year()
    # step interval between observation dates
    step_interval = int(round(steps_per_year / obs_per_year))

    # generate observation indices (exclude t=0, include final maturity)
    obs_indices: List[int] = list(range(step_interval, n_steps + 1, step_interval))

    # enforce that the last observation is exactly maturity
    if obs_indices[-1] != n_steps:
        obs_indices[-1] = n_steps

    obs_indices_arr = np.array(obs_indices, dtype=int)
    obs_times_arr = obs_indices_arr / steps_per_year

    return obs_indices_arr, obs_times_arr
