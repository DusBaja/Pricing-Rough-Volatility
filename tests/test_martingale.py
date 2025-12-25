import numpy as np
from pricer.core.models import Model
#Check E(DF(0,T)ST)=S0 within MC interval

def test_discounted_spot_is_martingale_under_curve_gbm():
    # Slightly upward-sloping zero curve (continuous-compounded)
    disc_times = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=float)
    disc_rates = np.array([0.02, 0.022, 0.024, 0.027, 0.030], dtype=float)

    m = Model(
        spot_process="GBM",
        rate_process="FLAT",
        vol_process="FLAT",
        s0=100.0,
        r0=0.02,    
        sigma=0.20,
        steps_per_year=252,
        n_paths=50_000,
        seed=123,
        disc_times=disc_times,
        disc_rates=disc_rates,
    )

    T = 2.0
    paths = m.simulate_paths(maturity_years=T, antithetic=True)
    ST = paths["S"][:, -1]
    DF = paths["df"][:, -1]

    x = DF * ST
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))

    
    assert abs(mean - m.s0) < 6.0 * se


def test_discounted_spot_is_martingale_under_curve_heston():
    disc_times = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=float)
    disc_rates = np.array([0.02, 0.022, 0.024, 0.027, 0.030], dtype=float)

    m = Model(
        spot_process="HESTON",
        rate_process="FLAT",
        vol_process="HESTON",
        s0=100.0,
        r0=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.5,
        rho_sv=-0.7,
        steps_per_year=252,
        n_paths=50_000,
        seed=123,
        disc_times=disc_times,
        disc_rates=disc_rates,
    )

    T = 2.0
    paths = m.simulate_paths(maturity_years=T, antithetic=True)
    ST = paths["S"][:, -1]
    DF = paths["df"][:, -1]

    x = DF * ST
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))

    assert abs(mean - m.s0) < 6.0 * se

def test_discounted_spot_martingale_under_curve_rbergomi():
    disc_times = np.array([0.0, 0.5, 1.0, 2.0, 5.0], dtype=float)
    disc_rates = np.array([0.02, 0.022, 0.024, 0.027, 0.03], dtype=float)

    m = Model(
        spot_process="GBM",
        vol_process="RBERGOMI",
        rate_process="FLAT",
        s0=100.0,
        r0=0.02,
        sigma=0.20,          # not used directly by rBergomi
        rough_H=0.1,
        rough_nu=1.0,        
        rough_rho=-0.7,      
        steps_per_year=252,
        n_paths=50_000,
        seed=123,
        disc_times=disc_times,
        disc_rates=disc_rates,
    )
    m._build_correlations()

    T = 2.0
    paths = m.simulate_paths(maturity_years=T, antithetic=True)
    ST = paths["S"][:, -1]
    DF = paths["df"][:, -1]

    x = DF * ST
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))

    assert abs(mean - m.s0) < 4.0 * se