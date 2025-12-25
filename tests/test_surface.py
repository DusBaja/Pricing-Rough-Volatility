import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pricer.market.surface import parse_sx5e_bbg_surface

surf = parse_sx5e_bbg_surface("data/raw/OptionsSX5E.xlsx",val_date="2025-12-15")

print(surf.head(20))
print("rows:", len(surf), "unique expiries:", surf["expiry"].nunique())
print("T range:", surf["T"].min(), surf["T"].max())
print("iv range:", surf["iv"].min(), surf["iv"].max())
print(surf.groupby(["expiry","cp"]).size())
