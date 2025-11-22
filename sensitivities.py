# sensitivities.py

def finite_diff_sensitivity(
    pricer_fn,
    base_params: dict,
    param_name: str,
    rel_bump: float = 0.01,
    central: bool = False,  # default: forward diff for speed
):
    """
    Generic bump-and-revalue sensitivity.

    pricer_fn(**params) -> (price, std_error)
    base_params : dict of parameters passed to pricer_fn
    param_name  : key of the parameter to bump
    rel_bump    : relative bump size (e.g. 0.01 = 1%)
    central     : if True, use central difference (2 evals),
                  otherwise forward difference (1 extra eval).
    """
    base_params = base_params.copy()
    x0 = base_params[param_name]

    # common random numbers
    seed0 = base_params.get("seed", 12345)
    base_params["seed"] = seed0

    # base price
    p0, _ = pricer_fn(**base_params)

    if not central:
        # forward diff
        x_plus = x0 * (1.0 + rel_bump)
        params_plus = base_params.copy()
        params_plus[param_name] = x_plus
        params_plus["seed"] = seed0

        p_plus, _ = pricer_fn(**params_plus)

        return (p_plus - p0) / (x_plus - x0)

    # central difference (slower, but more accurate)
    x_plus = x0 * (1.0 + rel_bump)
    x_minus = x0 * (1.0 - rel_bump)

    params_plus = base_params.copy()
    params_minus = base_params.copy()

    params_plus[param_name] = x_plus
    params_minus[param_name] = x_minus

    params_plus["seed"] = seed0
    params_minus["seed"] = seed0

    p_plus, _ = pricer_fn(**params_plus)
    p_minus, _ = pricer_fn(**params_minus)

    return (p_plus - p_minus) / (x_plus - x_minus)


def all_param_sensitivities(
    pricer_fn,
    base_params: dict,
    param_names,
    rel_bump=0.01,
    central=False,
):
    """
    Compute sensitivities for all params in param_names.
    Returns dict: {param_name: sensitivity}
    """
    sens = {}
    for name in param_names:
        sens[name] = finite_diff_sensitivity(
            pricer_fn,
            base_params,
            name,
            rel_bump=rel_bump,
            central=central,
        )
    return sens
