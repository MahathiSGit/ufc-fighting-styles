"""
Leakage + sanity tests for preprocessing/build_dataset.py.   Run:  python tests/test_no_leakage.py

1. Brute force : recompute pre-fight features by hand from strictly-earlier fights and compare.
2. Perturbation: change a fight's own result/stats -> that fight's features must NOT change,
                 but the fighter's NEXT fight's features must change.
3. Sanity       : balanced A/B assignment, no outcome-like columns among model features,
                 and the baselines a model has to beat.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
import build_dataset as bd

m = bd.load_master()
L, hist, fights, latest, P = bd.build_all(m)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. brute force
sample = hist[hist.date >= bd.MODEL_START].sample(60, random_state=1)
for r in sample.itertuples():
    past = L[(L.fighter_id == r.fighter_id) & (L.date < r.date)]
    assert r.n_fights == len(past), "n_fights mismatch"
    assert abs(r.win_rate - (past.wins.sum() + 2) / (len(past) + 4)) < 1e-9, "win_rate mismatch"
    mins = past.secs.sum() / 60
    exp_slpm = (past.sig_l.sum() + P["slpm"] * 15) / (mins + 15)
    assert abs(r.slpm - exp_slpm) < 1e-9, "slpm mismatch"
    exp_td = (past.td_l.sum() / max(past.td_a.sum(), 1e-9)) if False else (past.td_l.sum() + P["td_acc"] * 10) / (past.td_a.sum() + 10)
    assert abs(r.td_acc - exp_td) < 1e-9, "td_acc mismatch"
    if len(past):
        assert abs(r.layoff_days - (r.date - past.date.max()).days) < 1e-9, "layoff mismatch"
print("1. brute-force recomputation: OK (60 fighter-fights)")

# ---------------------------------------------------------------- 2. perturbation
cand = hist[(hist.date >= "2016-01-01")].copy()
cand = cand[cand.groupby("fighter_id").fight_id.transform("count") >= 3]
row = cand.sample(1, random_state=3).iloc[0]
fid, fighter = row.fight_id, row.fighter_id
m2 = m.copy()
i = m2.index[m2.fight_id == fid][0]
# flip the winner and inflate every stat of this one fight
m2.loc[i, "winner_id"] = m2.loc[i, "b_fighter_id"] if m2.loc[i, "winner_id"] == m2.loc[i, "r_fighter_id"] else m2.loc[i, "r_fighter_id"]
for c in [c for c in m2.columns if c.startswith(("r_total_", "b_total_"))]:
    m2.loc[i, c] = m2.loc[i, c] * 10 + 7
m2.loc[i, "method"] = "Submission"
_, hist2, _, _ , _ = bd.build_all(m2)

key = ["fight_id", "fighter_id"]
feat = bd.NUM_FEATS
a = hist.set_index(key)[feat]
b = hist2.set_index(key)[feat]
same_fight = a.loc[(fid, fighter)].fillna(-999).values == b.loc[(fid, fighter)].fillna(-999).values
assert same_fight.all(), "the fight's OWN features changed when its own result changed -> leakage!"
later = hist[(hist.fighter_id == fighter) & (hist.date > row.date)].sort_values("date")
nxt = later.iloc[0].fight_id
changed = (a.loc[(nxt, fighter)].fillna(-999).values != b.loc[(nxt, fighter)].fillna(-999).values).any()
assert changed, "next fight should reflect the changed result"
print(f"2. perturbation: OK (fight {fid}: own features unchanged, next fight {nxt} changed)")

# ---------------------------------------------------------------- 3. sanity
assert 0.47 < fights.meta_a_is_red.mean() < 0.53, "A/B assignment not balanced"
assert 0.47 < fights.a_wins.mean() < 0.53, "target not balanced"
cols = bd.feature_columns(fights)
bad = [c for c in cols if any(w in c.lower() for w in ("win_by", "winner", "result", "method", "target", "a_wins", "is_red"))]
assert not bad, f"suspicious feature columns: {bad}"
assert fights.fight_id.is_unique and (fights.a_id != fights.b_id).all()
print(f"3. sanity: OK ({len(cols)} model feature columns, A/B balanced, target mean {fights.a_wins.mean():.3f})")

# ---------------------------------------------------------------- baselines
print("\nBaselines to beat (accuracy):")
for s, g in fights.groupby("meta_split"):
    red = (g.a_wins == g.meta_a_is_red.astype(int)).mean()
    elo = ((g.diff_elo > 0).astype(int) == g.a_wins)[g.diff_elo != 0].mean()
    exp = ((g.diff_n_fights > 0).astype(int) == g.a_wins)[g.diff_n_fights != 0].mean()
    print(f"  {s:5} n={len(g):5} | always-red {red:.3f} | higher Elo {elo:.3f} | more experienced {exp:.3f} | coin {max(g.a_wins.mean(), 1-g.a_wins.mean()):.3f}")
print("\nALL TESTS PASSED")
