import math

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_forward_price(F: float, K: float, T: float, vol: float, df: float, cp: str) -> float:
    """Black price on forward F with discount df. cp in {'C','P'}."""
    if T <= 0:
        return df * max((F - K) if cp.upper() == "C" else (K - F), 0.0)
    if vol <= 0:
        return df * max((F - K) if cp.upper() == "C" else (K - F), 0.0)

    srt = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / srt
    d2 = d1 - srt

    if cp.upper() == "C":
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:
        return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))

def implied_vol_black(F: float, K: float, T: float, df: float, price: float, cp: str) -> float:
    """Implied vol via safe bisection. Returns vol in decimals."""
    #  lower bound
    intrinsic = df * max((F - K) if cp.upper() == "C" else (K - F), 0.0)
    price = max(price, intrinsic)

    lo, hi = 1e-6, 5.0  
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pmid = black_forward_price(F, K, T, mid, df, cp)
        if pmid > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
