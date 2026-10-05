"""Style labels shown in the app must equal the labels used in the style experiment.
Rebuild fighter states from fights before an event date, label them with style_of(), and compare to the
A_style/B_style stored in fight_features_style.csv.   Run: python tests/test_style_predictor.py"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
for p in ("preprocessing", "app"):
    sys.path.insert(0, str(ROOT / p))
import build_dataset as bd, style_predictor as sp

m = bd.load_master()
bundle = sp.load_style_assets(); A = bundle["assets"]
fs = pd.read_csv(ROOT / "data/processed/fight_features_style.csv", parse_dates=["meta_date"])
n = bad = 0
for D in ["2024-06-01", "2025-03-01", "2026-01-01"]:
    d0 = fs[fs.meta_date >= pd.Timestamp(D)].meta_date.min()
    latest = bd.build_all(m[m.date < d0])[3].set_index("fighter_id")
    for r in fs[fs.meta_date == d0].itertuples():
        for fid, want in ((r.a_id, r.A_style), (r.b_id, r.B_style)):
            if fid not in latest.index:
                continue
            got = sp.style_of(latest.loc[fid], A); n += 1
            bad += got != want
assert n >= 20 and bad == 0, f"{bad} of {n} style labels differ"
print(f"style labels in the app == labels in the experiment for {n} fighter-fights: OK")
