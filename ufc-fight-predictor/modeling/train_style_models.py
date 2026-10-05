"""
Train + save the two models behind the style-aware predictor, and the objects needed to label a fighter's style.

Reads : data/processed/fighter_history.csv, fight_features.csv
Writes: models/style_models.joblib   {skill, style, assets}

  skill  = logistic regression on skill differences only
  style  = same + style-cluster features (style presence differences + style-matchup edge)

The clusters, the matchup matrix and all hyper-parameters follow the evaluation protocol in
modeling/compare_models.py (fit on the training period; C chosen on the 2023 validation split).
Reported test metrics come from models trained on 2014-2022; the saved models are refit on all fights.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modeling"))
import style_clusters as sc  # noqa: E402
from compare_models import GRIDS, SKILL, STYLE_CLUSTER, make_model  # noqa: E402

PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"
SETS = {"skill": [f"diff_{c}" for c in SKILL], "style": [f"diff_{c}" for c in SKILL] + STYLE_CLUSTER}


def fit(C, X, y):
    return make_model("logreg", {"C": C}).fit(np.vstack([X, -X]), np.r_[y, 1 - y])


def proba(m, X):
    return np.clip(0.5 * (m.predict_proba(X)[:, 1] + 1 - m.predict_proba(-X)[:, 1]), 1e-6, 1 - 1e-6)


def main():
    hist = pd.read_csv(PROC / "fighter_history.csv", parse_dates=["date"])
    fights = pd.read_csv(PROC / "fight_features.csv", parse_dates=["meta_date"])
    f, _, scaler, km, order, _, (S, N) = sc.add_style_features(fights, hist)
    tr, va, te = (f[f.meta_split == s] for s in ("train", "val", "test"))

    out = {}
    for name, cols in SETS.items():
        X = {k: g[cols].fillna(0).values for k, g in (("train", tr), ("val", va), ("test", te))}
        y = {k: g.a_wins.values for k, g in (("train", tr), ("val", va), ("test", te))}
        C = min([p["C"] for p in GRIDS["logreg"]], key=lambda c: log_loss(y["val"], proba(fit(c, X["train"], y["train"]), X["val"])))
        pt = proba(fit(C, X["train"], y["train"]), X["test"])
        metrics = dict(acc=accuracy_score(y["test"], pt > .5), logloss=log_loss(y["test"], pt),
                       brier=brier_score_loss(y["test"], pt), n_test=len(pt))
        final = fit(C, f[cols].fillna(0).values, f.a_wins.values)           # refit on all fights
        out[name] = dict(model=final, columns=cols, C=C, test_metrics=metrics)
        print(f"{name:6} C={C:<6} test acc {metrics['acc']:.3f}  log-loss {metrics['logloss']:.3f}  brier {metrics['brier']:.3f}")

    out["assets"] = dict(scaler=scaler, km=km, order=order, S=S, N=N, K=sc.K, shrink=sc.SHRINK,
                         style_inputs=sc.STYLE_INPUTS, min_stat_fights=sc.MIN_STAT_FIGHTS, names=sc.NAME_ORDER)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(out, MODELS / "style_models.joblib")
    print("Saved models/style_models.joblib")


if __name__ == "__main__":
    main()
