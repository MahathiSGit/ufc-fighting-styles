"""
Step 4 - does style help beyond skill?

Reads : data/processed/fight_features_style.csv   (from modeling/style_clusters.py)
Writes: reports/model_comparison.csv, reports/figures/model_comparison.png

Design
  * Every feature is an A-minus-B difference, so the problem is exactly antisymmetric. We train on each
    fight twice (A/B swapped, target flipped) and average the prediction of a fight and its mirror image.
  * Feature sets:  skill only | skill + style clusters | skill + continuous style descriptors
  * Hyper-parameters are chosen on the 2023 validation split by log-loss. The 2024-2026 test split is
    scored once with the chosen settings. No feature or model choice was changed after seeing test scores.
  * Uncertainty: 2,000 bootstrap resamples of the test fights; differences are paired (same resamples).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC, REP = ROOT / "data" / "processed", ROOT / "reports"

SKILL = ["win_rate", "n_fights", "elo", "streak", "layoff_days", "age", "height_in", "reach_in",
         "sig_acc", "sig_def", "td_acc", "td_def", "sig_diff_pm", "kd15", "kda15", "ko_l_rate", "sub_l_rate"]
STYLE_CONT = ["slpm", "sapm", "head_share", "body_share", "leg_share", "dist_share", "clinch_share",
              "ground_share", "td15", "sub15", "ctrl_share", "ko_w_rate", "sub_w_rate", "dec_w_rate"]
STYLE_CLUSTER = ["diff_style_edge", "diff_style_unk"] + [f"diff_style_{k}" for k in range(5)]
FEATURE_SETS = {
    "Skill only": [f"diff_{c}" for c in SKILL],
    "Skill + style clusters": [f"diff_{c}" for c in SKILL] + STYLE_CLUSTER,
    "Skill + style descriptors": [f"diff_{c}" for c in SKILL] + [f"diff_{c}" for c in STYLE_CONT],
}
N_BOOT, SEED = 2000, 0


def make_model(kind, params):
    if kind == "logreg":
        return make_pipeline(StandardScaler(with_mean=False),
                             LogisticRegression(C=params["C"], fit_intercept=False, max_iter=5000))
    return HistGradientBoostingClassifier(learning_rate=0.03, max_depth=params["depth"], max_iter=params["iters"],
                                          min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                                          random_state=SEED)


GRIDS = {"logreg": [{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)],
         "hgb": [{"depth": d, "iters": i} for d in (2, 3) for i in (100, 200, 400)]}


def fit_predict(kind, params, Xtr, ytr, Xte):
    """Train on fights + mirror images; predict as average of a fight and its mirrored version."""
    m = make_model(kind, params).fit(np.vstack([Xtr, -Xtr]), np.r_[ytr, 1 - ytr])
    return 0.5 * (m.predict_proba(Xte)[:, 1] + 1 - m.predict_proba(-Xte)[:, 1])


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(acc=accuracy_score(y, p > 0.5), logloss=log_loss(y, p), brier=brier_score_loss(y, p))


def main():
    df = pd.read_csv(PROC / "fight_features_style.csv", parse_dates=["meta_date"])
    tr, va, te = (df[df.meta_split == s] for s in ("train", "val", "test"))
    y = {n: g.a_wins.values for n, g in (("train", tr), ("val", va), ("test", te))}
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}  (style known for both: test {te.style_known_both.mean():.0%})")

    results, test_pred = [], {}
    for fs_name, cols in FEATURE_SETS.items():
        X = {n: g[cols].fillna(0).values for n, g in (("train", tr), ("val", va), ("test", te))}
        for kind in ("logreg", "hgb"):
            best = min(GRIDS[kind], key=lambda p: log_loss(y["val"], np.clip(fit_predict(kind, p, X["train"], y["train"], X["val"]), 1e-6, 1 - 1e-6)))
            pv = fit_predict(kind, best, X["train"], y["train"], X["val"])
            pt = fit_predict(kind, best, X["train"], y["train"], X["test"])
            name = f"{fs_name} | {'Logistic' if kind == 'logreg' else 'GradBoost'}"
            test_pred[name] = pt
            mv, mt = metrics(y["val"], pv), metrics(y["test"], pt)
            results.append(dict(model=name, feature_set=fs_name, algo=kind, params=str(best),
                                **{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}))
            print(f"{name:42} {str(best):28} val acc {mv['acc']:.3f} ll {mv['logloss']:.3f} | test acc {mt['acc']:.3f} ll {mt['logloss']:.3f}")

    # baselines on test
    red = (te.a_wins.values == te.meta_a_is_red.astype(int).values).mean()
    elo_p = 1 / (1 + 10 ** (-te.diff_elo.values / 400))
    base = metrics(y["test"], elo_p)
    print(f"\nBaselines (test): always-red acc {red:.3f} | Elo-only acc {base['acc']:.3f} logloss {base['logloss']:.3f}")

    # paired bootstrap on test
    rng = np.random.default_rng(SEED)
    n = len(te)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    yt = y["test"]
    def per_sample(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return (p > 0.5) == yt, -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
    ps = {k: per_sample(v) for k, v in test_pred.items()}
    rows = []
    for algo in ("Logistic", "GradBoost"):
        ref = f"Skill only | {algo}"
        for name in test_pred:
            if not name.endswith(algo):
                continue
            acc_b, ll_b = ps[name][0][idx].mean(1), ps[name][1][idx].mean(1)
            d_acc = acc_b - ps[ref][0][idx].mean(1)
            d_ll = ll_b - ps[ref][1][idx].mean(1)
            rows.append(dict(model=name,
                             acc_lo=np.percentile(acc_b, 2.5), acc_hi=np.percentile(acc_b, 97.5),
                             ll_lo=np.percentile(ll_b, 2.5), ll_hi=np.percentile(ll_b, 97.5),
                             d_acc_vs_skill=d_acc.mean(), d_acc_lo=np.percentile(d_acc, 2.5), d_acc_hi=np.percentile(d_acc, 97.5),
                             d_ll_vs_skill=d_ll.mean(), d_ll_lo=np.percentile(d_ll, 2.5), d_ll_hi=np.percentile(d_ll, 97.5)))
    boot = pd.DataFrame(rows)

    # secondary view: only fights where both fighters have a known style
    both = te.style_known_both.values == 1
    sub = []
    for name, p in test_pred.items():
        mm = metrics(yt[both], p[both])
        sub.append(dict(model=name, n=int(both.sum()), sub_acc=mm["acc"], sub_logloss=mm["logloss"]))
    sub = pd.DataFrame(sub)

    out = pd.DataFrame(results).merge(boot, on="model").merge(sub, on="model")
    REP.mkdir(exist_ok=True)
    out.to_csv(REP / "model_comparison.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\nTest set, 95% bootstrap CIs; 'd_*' = change vs Skill only (same algorithm):")
    show = out[["model", "test_acc", "acc_lo", "acc_hi", "test_logloss", "ll_lo", "ll_hi", "d_acc_vs_skill", "d_acc_lo", "d_acc_hi", "d_ll_vs_skill", "d_ll_lo", "d_ll_hi"]]
    print(show.round(3).to_string(index=False))
    print("\nFights where both styles are known (n=%d):" % both.sum())
    print(sub.round(3).to_string(index=False))
    make_figure(out, red, base)


def make_figure(out, red, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (REP / "figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = [m.replace(" | ", "\n") for m in out.model]
    y = np.arange(len(out))
    for ax, col, lo, hi, title, ref in [(axes[0], "test_acc", "acc_lo", "acc_hi", "Test accuracy (higher is better)", red),
                                        (axes[1], "test_logloss", "ll_lo", "ll_hi", "Test log-loss (lower is better)", base["logloss"])]:
        ax.errorbar(out[col], y, xerr=[out[col] - out[lo], out[hi] - out[col]], fmt="o", capsize=3)
        ax.axvline(ref, color="grey", ls="--", lw=1)
        ax.text(ref, 1.0, "always-red" if col == "test_acc" else "Elo-only", transform=ax.get_xaxis_transform(), color="grey", fontsize=8, ha="center", va="bottom")
        ax.set_yticks(y, labels, fontsize=8); ax.invert_yaxis(); ax.set_title(title, pad=14)
    fig.suptitle("Skill vs skill + style - 95% bootstrap intervals on the 2024-2026 test set")
    fig.tight_layout(); fig.savefig(REP / "figures" / "model_comparison.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
