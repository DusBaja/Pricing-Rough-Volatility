# observation.py
import numpy as np
from enum import Enum, auto

class ObservationFrequency(Enum):
    ANNUAL = auto()
    MONTHLY = auto()
    DAILY = auto()

def get_observation_indices(maturity_years: float, steps_per_year: int, freq: ObservationFrequency):
    """
    Returns time indices at which autocall observations occur.
    Always includes maturity.
    """
    n_steps = int(maturity_years * steps_per_year)

    if freq == ObservationFrequency.ANNUAL:
        step_gap = steps_per_year
    elif freq == ObservationFrequency.MONTHLY:
        step_gap = max(1, steps_per_year // 12)
    elif freq == ObservationFrequency.DAILY:
        step_gap = 1
    else:
        raise ValueError("Unknown observation frequency")

    obs_indices = np.arange(step_gap, n_steps + 1, step_gap, dtype=int)
    obs_indices[-1] = n_steps
    obs_indices = np.unique(obs_indices)
    return obs_indices, n_steps
