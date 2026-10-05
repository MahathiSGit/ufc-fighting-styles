"""
The predictor must give a fighter the SAME features the training data gave them.
Test: rebuild fighter states using only fights before an event date D, run the predictor's feature code for the
fights on D, and compare with the pre-fight features stored in fight_features.csv.   Run: python tests/test_predictor.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocessing")); sys.path.insert(0, str(ROOT / "app"))
import build_dataset as bd, predictor as pr

m = bd.load_master()
_, _, fights, _, _ = bd.build_all(m)
checked = 0
for D in ["2024-06-01", "2025-03-01", "2026-01-01"]:
    D = pd.Timestamp(D)
    d0 = fights[fights.meta_date >= D].meta_date.min()          # first event on/after D
    day = fights[fights.meta_date == d0]
    _, _, _, latest, _ = bd.build_all(m[m.date < d0])
    latest = latest.set_index("fighter_id")
    for r in day.itertuples():
        if r.a_id not in latest.index or r.b_id not in latest.index:
            continue                                   # debut: no history to compare
        for side, fid in (("A", r.a_id), ("B", r.b_id)):
            got = pr.features_for(latest.loc[fid], d0)
            for c in bd.NUM_FEATS:
                want = getattr(r, f"{side}_{c}")
                if pd.isna(want) and pd.isna(got[c]):
                    continue
                assert abs(got[c] - want) < 1e-6, f"{c} mismatch for {fid} on {d0:%Y-%m-%d}: {got[c]} vs {want}"
        checked += 1
assert checked >= 10, checked
print(f"predictor features == training features for {checked} real fights: OK")
