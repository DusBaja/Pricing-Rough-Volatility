import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def build_xi0_from_atm(surf: pd.DataFrame):
    out_t, out_iv = [], []

    for expiry, g in surf.groupby("expiry"):
        T = float(g["T"].iloc[0])
        gF = g["F_bbg"].dropna()
        if len(gF) == 0:
            continue
        F = float(gF.iloc[0])

        strikes = np.sort(g["K"].unique())
        K_atm = float(strikes[np.argmin(np.abs(strikes - F))])

        ivs = g.loc[g["K"] == K_atm, "iv"].dropna().values
        if len(ivs) == 0:
            continue
        iv_atm = float(np.mean(ivs))

        out_t.append(T)
        out_iv.append(iv_atm)

        print("ATM pick:", expiry, "T", T, "F", F, "K_atm", K_atm, "iv_atm", iv_atm)

    T = np.array(out_t, dtype=float)
    iv = np.array(out_iv, dtype=float)
    idx = np.argsort(T)
    T, iv = T[idx], iv[idx]

    # total variance w(T)=iv^2*T
    w = iv * iv * T

    # piecewise-constant forward variance levels
    T0 = np.concatenate([[0.0], T])
    w0 = np.concatenate([[0.0], w])

    fwd_var = (w0[1:] - w0[:-1]) / (T0[1:] - T0[:-1])

    # represent xi0(t) as piecewise constant by giving knot points T and values fwd_var, (our _xi0_at_times uses linear interpolation)
    xi0_times = T
    xi0_values = fwd_var

    return xi0_times, xi0_values

