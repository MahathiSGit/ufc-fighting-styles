"""
Fight Predictor - compare two fighters and estimate who wins.

Uses data/processed/fighter_latest_state.csv (each fighter's record/stats AFTER their latest fight, built from
past fights only) and the model in models/fight_model.joblib. For a fight on date D, age and layoff are
measured at D, exactly as they are in the training data.

Command line:
    python app/predictor.py "Islam Makhachev" "Charles Oliveira"
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocessing"))
import build_dataset as bd  # noqa: E402

LABELS = {  # friendly names for the comparison table
    "elo": "Elo rating", "win_rate": "Win rate (smoothed)", "n_fights": "UFC fights", "streak": "Win(+)/loss(-) streak",
    "age": "Age", "height_in": "Height (in)", "reach_in": "Reach (in)", "layoff_days": "Days since last fight",
    "sig_acc": "Striking accuracy", "sig_def": "Striking defense", "slpm": "Sig. strikes landed /min",
    "sapm": "Sig. strikes absorbed /min", "sig_diff_pm": "Strike differential /min", "td15": "Takedowns /15 min",
    "td_acc": "Takedown accuracy", "td_def": "Takedown defense", "sub15": "Sub attempts /15 min",
    "ctrl_share": "Control time share", "kd15": "Knockdowns /15 min", "kda15": "Knockdowns taken /15 min",
    "ko_w_rate": "KO/TKO win rate", "sub_w_rate": "Submission win rate", "dec_w_rate": "Decision win rate",
    "ko_l_rate": "KO/TKO loss rate", "sub_l_rate": "Submission loss rate", "head_share": "Strikes to head",
    "body_share": "Strikes to body", "leg_share": "Strikes to legs", "dist_share": "Strikes at distance",
    "clinch_share": "Strikes in clinch", "ground_share": "Strikes on ground", "n_stat_fights": "Fights with detailed stats",
}


def load_assets():
    state = pd.read_csv(ROOT / "data" / "processed" / "fighter_latest_state.csv", parse_dates=["dob", "last_fight_date"])
    bundle = joblib.load(ROOT / "models" / "fight_model.joblib")
    return state, bundle


def fighter_options(state):
    """Labels for a dropdown, unique even when two fighters share a name."""
    s = state.sort_values("name")
    lab = [f"{r.name} ({int(r.n_fights)} UFC fights, last fought {r.last_fight_date:%b %Y})" for r in s.itertuples()]
    return dict(zip(lab, s.fighter_id))


def features_for(row, fight_date):
    """The same pre-fight features the model was trained on, for one fighter on a given date."""
    f = {c: row[c] for c in bd.DERIVED}
    f["elo"], f["streak"] = row["elo"], row["streak"]
    f["height_in"], f["reach_in"] = row["height_in"], row["reach_in"]
    f["age"] = (pd.Timestamp(fight_date) - row["dob"]).days / 365.25 if pd.notna(row["dob"]) else np.nan
    f["layoff_days"] = (pd.Timestamp(fight_date) - row["last_fight_date"]).days
    return f


def _p(model, x):
    return 0.5 * (model.predict_proba(x)[:, 1] + 1 - model.predict_proba(-x)[:, 1])


def predict_fight(a_row, b_row, bundle, fight_date=None):
    fight_date = pd.Timestamp(fight_date or pd.Timestamp.today().normalize())
    fa, fb = features_for(a_row, fight_date), features_for(b_row, fight_date)
    cols = bundle["columns"]
    x = np.array([[fa[c[5:]] - fb[c[5:]] for c in cols]], dtype=float)
    x = np.nan_to_num(x, nan=0.0)                         # same neutral fill used in training
    model = bundle["model"]
    p = float(_p(model, x)[0])
    logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    # what pushes the prediction? effect of setting one feature difference back to 0 (= "no difference")
    drivers = []
    for j, c in enumerate(cols):
        x0 = x.copy(); x0[0, j] = 0.0
        drivers.append((c[5:], float(logit(p) - logit(_p(model, x0)[0]))))
    drivers = sorted(drivers, key=lambda t: -abs(t[1]))
    table = pd.DataFrame({"stat": [LABELS.get(c[5:], c[5:]) for c in cols],
                          a_row["name"]: [fa[c[5:]] for c in cols], b_row["name"]: [fb[c[5:]] for c in cols]})
    return dict(p_a=p, p_b=1 - p, drivers=drivers, table=table, date=fight_date)


def find(state, name):
    hits = state[state["name"].str.lower() == name.lower()]
    if hits.empty:
        hits = state[state["name"].str.lower().str.contains(name.lower(), regex=False)]
    if hits.empty:
        raise SystemExit(f"No fighter matching '{name}'")
    return hits.sort_values("last_fight_date").iloc[-1]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python app/predictor.py "Fighter A" "Fighter B"')
    state, bundle = load_assets()
    a, b = find(state, sys.argv[1]), find(state, sys.argv[2])
    r = predict_fight(a, b, bundle)
    print(f"\n{a['name']}  vs  {b['name']}   (as of {r['date']:%Y-%m-%d}, model: {bundle['name']})")
    print(f"  {a['name']}: {r['p_a']:.1%}\n  {b['name']}: {r['p_b']:.1%}\n")
    print("Biggest factors (positive favours", a["name"] + "):")
    for c, v in r["drivers"][:5]:
        print(f"  {LABELS.get(c, c):28} {v:+.2f}")
    pd.set_option("display.width", 120)
    print("\n", r["table"].round(2).to_string(index=False))
