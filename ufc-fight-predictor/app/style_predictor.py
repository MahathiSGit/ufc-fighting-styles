"""
Style-aware fight prediction: labels each fighter's style, looks up the style-vs-style matchup edge,
and predicts with a skill-only model and a skill + style model side by side.

    python app/style_predictor.py "Islam Makhachev" "Charles Oliveira"
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predictor as pr  # noqa: E402

ROOT = HERE.parent


def load_style_assets():
    return joblib.load(ROOT / "models" / "style_models.joblib")


def style_of(row, a):
    """Style id (0..K-1) from the fighter's pre-fight stats, or -1 if there is too little history."""
    vals = row[a["style_inputs"]].astype(float)
    if row["n_stat_fights"] < a["min_stat_fights"] or vals.isna().any():
        return -1
    raw = a["km"].predict(a["scaler"].transform(pd.DataFrame([vals.values], columns=a["style_inputs"])))[0]
    return int(a["order"][raw])


def matchup(sa, sb, a):
    """Historical edge (percentage points above/below the Elo expectation) of style sa against style sb."""
    if sa < 0 or sb < 0 or sa == sb:
        return None
    n = int(a["N"][sa, sb])
    return dict(edge_pp=100 * a["S"][sa, sb] / n if n else 0.0, n=n,
                used=float(a["S"][sa, sb] / (n + a["shrink"])))


def _vector(cols, fa, fb, sa, sb, a):
    mu = matchup(sa, sb, a)
    v = []
    for c in cols:
        if c == "diff_style_edge":
            v.append(mu["used"] if mu else 0.0)
        elif c == "diff_style_unk":
            v.append(float(sa < 0) - float(sb < 0))
        elif c.startswith("diff_style_"):
            k = int(c.rsplit("_", 1)[1])
            v.append(float(sa == k) - float(sb == k))
        else:
            v.append(fa[c[5:]] - fb[c[5:]])
    return np.nan_to_num(np.array([v], dtype=float), nan=0.0)


def _p(m, x):
    return float(0.5 * (m.predict_proba(x)[0, 1] + 1 - m.predict_proba(-x)[0, 1]))


def label(c, names):
    if c == "diff_style_edge":
        return "Style matchup edge"
    if c == "diff_style_unk":
        return "Style unknown (too little history)"
    if c.startswith("diff_style_"):
        return f"Style: {names[int(c.rsplit('_', 1)[1])]}"
    return pr.LABELS.get(c[5:], c[5:])


def predict_with_style(a_row, b_row, bundle, fight_date=None):
    A = bundle["assets"]
    d = pd.Timestamp(fight_date or pd.Timestamp.today().normalize())
    fa, fb = pr.features_for(a_row, d), pr.features_for(b_row, d)
    sa, sb = style_of(a_row, A), style_of(b_row, A)
    out = dict(style_a=sa, style_b=sb, matchup=matchup(sa, sb, A), date=d)
    for key in ("skill", "style"):
        cols, m = bundle[key]["columns"], bundle[key]["model"]
        x = _vector(cols, fa, fb, sa, sb, A)
        out[f"p_{key}"] = _p(m, x)
        if key == "style":
            lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
            dr = []
            for j, c in enumerate(cols):
                x0 = x.copy(); x0[0, j] = 0.0
                dr.append((label(c, A["names"]), float(lg(out["p_style"]) - lg(_p(m, x0)))))
            out["drivers"] = sorted(dr, key=lambda t: -abs(t[1]))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python app/style_predictor.py "Fighter A" "Fighter B"')
    state, _ = pr.load_assets()
    bundle = load_style_assets()
    a, b = pr.find(state, sys.argv[1]), pr.find(state, sys.argv[2])
    r = predict_with_style(a, b, bundle)
    names = bundle["assets"]["names"]
    nm = lambda s: names[s] if s >= 0 else "Unknown (fewer than 3 fights with detailed stats)"
    print(f"\n{a['name']} [{nm(r['style_a'])}]  vs  {b['name']} [{nm(r['style_b'])}]")
    print(f"  Skill-only model     : {a['name']} {r['p_skill']:.1%}")
    print(f"  Skill + style model  : {a['name']} {r['p_style']:.1%}")
    if r["matchup"]:
        m = r["matchup"]
        print(f"  Style matchup: {names[r['style_a']]} vs {names[r['style_b']]} historically {m['edge_pp']:+.1f} points "
              f"vs Elo expectation (n={m['n']} fights, 2014-2022)")
    print("  Top factors (skill + style model):")
    for c, v in r["drivers"][:5]:
        print(f"    {c:36} {v:+.2f}")
