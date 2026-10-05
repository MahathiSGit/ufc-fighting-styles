"""
Step 3 (base project) - Logistic Regression vs Random Forest vs Gradient Boosting.

Reads : data/processed/fight_features.csv           (from preprocessing/build_dataset.py)
Writes: reports/model_results.csv, reports/figures/*.png, models/fight_model.joblib

Protocol
  * Features  : every diff_ column = (fighter A) minus (fighter B), computed from PRE-fight history only.
  * Splits    : train 2014-2022 | validation 2023 | test 2024-2026 (chronological, never shuffled).
  * Symmetry  : each fight is trained twice (A/B swapped, target flipped) and predicted as the average of
                a fight and its mirror image, so the model cannot learn anything from the A/B labels.
  * Tuning    : hyper-parameters chosen on the validation split by log-loss; test is scored once.
  * Deployment: the model saved for the demo app is refit on ALL fights with the chosen settings.
                Reported metrics always come from the train-only model.
"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocessing"))
import build_dataset as bd  # noqa: E402

PROC, REP, FIG, MODELS = ROOT / "data" / "processed", ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"
FEATURES = [f"diff_{c}" for c in bd.NUM_FEATS]
SEED, N_BOOT = 0, 2000
NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "gb": "Gradient Boosting"}

GRIDS = {
    "logreg": [{"C": c} for c in (0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)],
    "rf": [{"depth": d, "leaf": l} for d in (4, 6, 8, 12) for l in (10, 20, 40)],
    "gb": [{"depth": d, "iters": i} for d in (2, 3) for i in (100, 200, 400)],
}


def make(kind, p):
    if kind == "logreg":
        return make_pipeline(StandardScaler(with_mean=False),
                             LogisticRegression(C=p["C"], fit_intercept=False, max_iter=5000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=p["depth"], min_samples_leaf=p["leaf"],
                                      max_features="sqrt", n_jobs=-1, random_state=SEED)
    return HistGradientBoostingClassifier(learning_rate=0.03, max_depth=p["depth"], max_iter=p["iters"],
                                          min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                                          random_state=SEED)


def fit(kind, p, X, y):
    return make(kind, p).fit(np.vstack([X, -X]), np.r_[y, 1 - y])


def proba(est, X):
    return 0.5 * (est.predict_proba(X)[:, 1] + 1 - est.predict_proba(-X)[:, 1])


def ll(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


def main():
    df = pd.read_csv(PROC / "fight_features.csv", parse_dates=["meta_date"])
    tr, va, te = (df[df.meta_split == s] for s in ("train", "val", "test"))
    X = {n: g[FEATURES].fillna(0).values for n, g in (("train", tr), ("val", va), ("test", te))}
    y = {n: g.a_wins.values for n, g in (("train", tr), ("val", va), ("test", te))}
    print(f"{len(FEATURES)} features | train {len(tr)} | val {len(va)} | test {len(te)}")

    fitted, test_p, val_p, chosen = {}, {}, {}, {}
    for kind in NAMES:
        best = min(GRIDS[kind], key=lambda p: ll(y["val"], proba(fit(kind, p, X["train"], y["train"]), X["val"])))
        est = fit(kind, best, X["train"], y["train"])
        fitted[kind], chosen[kind] = est, best
        val_p[kind], test_p[kind] = proba(est, X["val"]), proba(est, X["test"])
        print(f"{NAMES[kind]:20} chosen {best} | val acc {accuracy_score(y['val'], val_p[kind] > .5):.3f} "
              f"ll {ll(y['val'], val_p[kind]):.3f}")

    # ---------------- test evaluation with bootstrap CIs
    yt, n = y["test"], len(y["test"])
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    red = (te.a_wins.values == te.meta_a_is_red.astype(int).values)
    elo_p = 1 / (1 + 10 ** (-te.diff_elo.values / 400))
    rows = []
    allp = {**{NAMES[k]: v for k, v in test_p.items()}, "Elo only": elo_p}
    for name, p in allp.items():
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        correct = ((pc > .5) == yt).astype(float)
        loss = -(yt * np.log(pc) + (1 - yt) * np.log(1 - pc))
        ab, lb = correct[idx].mean(1), loss[idx].mean(1)
        aucb = np.array([roc_auc_score(yt[i], pc[i]) for i in idx[:500]])
        rows.append(dict(model=name, acc=correct.mean(), acc_lo=np.percentile(ab, 2.5), acc_hi=np.percentile(ab, 97.5),
                         logloss=loss.mean(), ll_lo=np.percentile(lb, 2.5), ll_hi=np.percentile(lb, 97.5),
                         brier=brier_score_loss(yt, pc), auc=roc_auc_score(yt, pc),
                         auc_lo=np.percentile(aucb, 2.5), auc_hi=np.percentile(aucb, 97.5)))
    res = pd.DataFrame(rows)
    rb = red[idx].mean(1)
    res.loc[len(res)] = dict(model="Always pick red corner", acc=red.mean(), acc_lo=np.percentile(rb, 2.5), acc_hi=np.percentile(rb, 97.5))
    REP.mkdir(exist_ok=True)
    res.to_csv(REP / "model_results.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nTEST set (2024-2026), 95% bootstrap intervals")
    print(res.round(3).to_string(index=False))

    best_kind = min(NAMES, key=lambda k: ll(y["val"], val_p[k]))     # selected on validation, not test
    print(f"\nBest on validation log-loss: {NAMES[best_kind]}")
    make_figures(res, te, X, y, fitted, test_p, best_kind, red.mean())

    # ---------------- deployment model: refit on everything with the chosen settings
    Xall, yall = np.vstack([X["train"], X["val"], X["test"]]), np.r_[y["train"], y["val"], y["test"]]
    final = fit(best_kind, chosen[best_kind], Xall, yall)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(dict(model=final, columns=FEATURES, kind=best_kind, name=NAMES[best_kind], params=chosen[best_kind],
                     trained_through=str(df.meta_date.max().date()), n_train=len(yall),
                     test_metrics=res[res.model == NAMES[best_kind]].iloc[0].to_dict()), MODELS / "fight_model.joblib")
    print("Saved models/fight_model.joblib (refit on all fights)")


def make_figures(res, te, X, y, fitted, test_p, best_kind, red_acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    yt = y["test"]
    # 1. accuracy comparison
    r = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.errorbar(r.acc, range(len(r)), xerr=[r.acc - r.acc_lo, r.acc_hi - r.acc], fmt="o", capsize=3)
    ax.set_yticks(range(len(r)), r.model); ax.axvline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("Test accuracy (95% bootstrap interval)"); ax.set_title("Model comparison on 2024-2026 fights")
    fig.tight_layout(); fig.savefig(FIG / "accuracy_comparison.png", dpi=160); plt.close(fig)
    # 2. ROC
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for k, p in test_p.items():
        fpr, tpr, _ = roc_curve(yt, p)
        ax.plot(fpr, tpr, label=f"{NAMES[k]} (AUC {roc_auc_score(yt, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8); ax.set_title("ROC curves (test)")
    fig.tight_layout(); fig.savefig(FIG / "roc_curves.png", dpi=160); plt.close(fig)
    # 3. calibration of best model
    p = test_p[best_kind]
    bins = pd.qcut(p, 8, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": yt, "b": bins}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0.3, 0.8], [0.3, 0.8], "k:", lw=1); ax.plot(g.p, g.y, "o-")
    for _, row in g.iterrows():
        ax.annotate(f"n={int(row.n)}", (row.p, row.y), fontsize=7, xytext=(3, -10), textcoords="offset points")
    ax.set_xlabel("Predicted probability fighter A wins"); ax.set_ylabel("Observed win rate")
    ax.set_title(f"Calibration - {NAMES[best_kind]} (test)"); fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=160); plt.close(fig)
    # 4. confusion matrix
    pred = (p > .5).astype(int)
    cm = np.array([[((yt == i) & (pred == j)).sum() for j in (0, 1)] for i in (0, 1)])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["B wins", "A wins"]); ax.set_yticks([0, 1], ["B won", "A won"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix - {NAMES[best_kind]}")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=160); plt.close(fig)
    # 5. permutation importance on VALIDATION data (test never used for explanation)
    est, Xv, yv = fitted[best_kind], X["val"], y["val"]
    base, rng, imp = ll(yv, proba(est, Xv)), np.random.default_rng(SEED), {}
    for j, c in enumerate(FEATURES):
        d = []
        for _ in range(10):
            Xp = Xv.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); d.append(ll(yv, proba(est, Xp)) - base)
        imp[c] = np.mean(d)
    top = pd.Series(imp).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh([c[len("diff_"):] for c in top.index], top.values)
    ax.set_xlabel("Increase in validation log-loss when the feature is shuffled")
    ax.set_title(f"Most useful features - {NAMES[best_kind]}"); fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=160); plt.close(fig)
    pd.Series(imp).sort_values(ascending=False).to_csv(REP / "feature_importance.csv", header=["perm_importance_logloss"])
    # 6. accuracy by experience
    mn = np.minimum(te.A_n_fights.values, te.B_n_fights.values)
    lab = pd.cut(mn, [-1, 2, 5, 10, 100], labels=["0-2", "3-5", "6-10", "11+"])
    acc = pd.DataFrame({"lab": lab, "ok": (pred == yt)}).groupby("lab", observed=True).ok.agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar(acc.index.astype(str), acc["mean"])
    for i, (m, n) in enumerate(zip(acc["mean"], acc["size"])):
        ax.text(i, m + 0.01, f"{m:.2f}\nn={n}", ha="center", fontsize=8)
    ax.set_ylim(0.4, 0.8); ax.set_xlabel("Fewer prior UFC fights of the two fighters"); ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy by fighter experience"); fig.tight_layout()
    fig.savefig(FIG / "accuracy_by_experience.png", dpi=160); plt.close(fig)
    print("Figures written to reports/figures/")


if __name__ == "__main__":
    main()
