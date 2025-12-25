import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Optional

MAT_PAT = re.compile(r"^\s*\d{1,2}-[A-Za-z]{3}-\d{2}")
FWD_PAT = re.compile(r"FwdI\s*([0-9]+(?:\.[0-9]+)?)")
R_PAT   = re.compile(r"\bR\s*([0-9]+(?:\.[0-9]+)?)")  

def parse_sx5e_bbg_surface(
    path: str,
    sheet: str = "Worksheet",
    val_date: Optional[str] = None,  # e.g. "2025-12-15"
) -> pd.DataFrame:
    """
    Parse Bloomberg-style SX5E option sheet with Calls block (0..6) and Puts block (7..13),
    using VIM as implied vol mid (in percent).
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)


    header_row = None
    for i in range(0, 10):
        row = raw.iloc[i].astype(str).str.strip().str.upper().tolist()
        if "STRIKE" in row and "VIM" in row:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find header row containing STRIKE and VIM")

    hdr = raw.iloc[header_row].astype(str).str.strip().str.upper().tolist()

 
    call_strike_col = 0
    call_vim_col = 5
    put_strike_col = 7
    put_vim_col = 12

    if hdr[call_vim_col] != "VIM" or hdr[put_vim_col] != "VIM":
        raise ValueError(f"Unexpected VIM locations: hdr[5]={hdr[5]}, hdr[12]={hdr[12]}")

    # valuation date
    if val_date is None:
        # fallback: infer from first expiry line "(4j)" is not reliable
        raise ValueError("Please provide val_date='YYYY-MM-DD' (Bloomberg valuation date).")

    val_dt = pd.Timestamp(val_date)

    rows = []
    current_expiry = None
    current_fwd = None
    current_r = None

    for i in range(header_row + 1, len(raw)):
        v0 = raw.iat[i, 0]
        s0 = "" if pd.isna(v0) else str(v0).strip()

        # maturity header row
        if MAT_PAT.match(s0):
            mdate = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{2})", s0)
            if not mdate:
                raise ValueError(f"Could not parse expiry date from header: {s0!r}")
            exp_str = mdate.group(1)
            current_expiry = datetime.strptime(exp_str, "%d-%b-%y").date()

            mf = FWD_PAT.search(s0)
            current_fwd = float(mf.group(1)) if mf else np.nan

            mr = R_PAT.search(s0)
            current_r = float(mr.group(1)) / 100.0 if mr else np.nan
            continue

        if current_expiry is None:
            continue

    
        k = raw.iat[i, call_strike_col]
        if pd.isna(k):
            continue
        try:
            K = float(k)
        except Exception:
            continue

        
        c = raw.iat[i, call_vim_col]
        p = raw.iat[i, put_vim_col]

        call_iv = np.nan
        put_iv = np.nan

        if not pd.isna(c):
            c = float(c)
            if c != 0:
                call_iv = c / 100.0

        if not pd.isna(p):
            p = float(p)
            if p != 0:
                put_iv = p / 100.0

        
        exp_ts = pd.Timestamp(current_expiry)
        T = (exp_ts - val_dt).days / 365.0

        if call_iv == call_iv:  
            rows.append(
                dict(expiry=current_expiry, T=T, K=K, iv=call_iv, cp="C", F_bbg=current_fwd, r_bbg=current_r)
            )
        if put_iv == put_iv:
            rows.append(
                dict(expiry=current_expiry, T=T, K=K, iv=put_iv, cp="P", F_bbg=current_fwd, r_bbg=current_r)
            )

    surf = pd.DataFrame(rows)

    
    surf = surf[surf["T"] > 0].copy()

    return surf
