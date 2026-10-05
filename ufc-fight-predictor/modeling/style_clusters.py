"""
Step 3 - fighter styles (clustering) + style-vs-style analysis.

Reads : data/processed/fighter_history.csv, fight_features.csv   (from preprocessing/build_dataset.py)
Writes: data/processed/fight_features_style.csv   fight table + style columns + style-matchup edge
        data/processed/fighter_styles.csv         style label of every fighter at every fight
        data/processed/style_profiles.csv         what each style looks like
        reports/figures/style_profiles.png, style_matrix.png

Leakage rules
  * A fighter's style at a fight is computed from their PRE-fight history only (fighter_history.csv).
  * The scaler + KMeans are fit on TRAIN-period data only (fights before 2023).
  * Fighters with fewer than MIN_STAT_FIGHTS previous fights with round data are labelled "Unknown".
  * The style-matchup "edge" (how much a style over/under-performs its Elo expectation against another
    style) is estimated from train fights only; for train rows it is leave-one-out, so a fight's own
    result never feeds its own feature.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocessing"))
import build_dataset as bd  # noqa: E402

PROC, FIG = ROOT / "data" / "processed", ROOT / "reports" / "figures"
STYLE_INPUTS = ["slpm", "head_share", "leg_share", "clinch_share", "ground_share", "td15", "sub15", "ctrl_share"]
MIN_STAT_FIGHTS = 3
K = 5
SHRINK = 30.0            # pseudo-fights pulling each matchup edge toward 0
SEED = 0
UNKNOWN = -1
NAME_ORDER = ["Volume striker", "Leg-kick striker", "Clinch grinder", "Wrestler / controller", "Submission hunter"]


def fit_style_model(hist):
    """Fit scaler + KMeans on each fighter's latest pre-fight state inside the training period."""
    tr = hist[(hist.date < bd.VAL_START) & (hist.n_stat_fights >= MIN_STAT_FIGHTS)]
    snap = tr.sort_values("date").groupby("fighter_id").tail(1)
    scaler = StandardScaler().fit(snap[STYLE_INPUTS])
    km = KMeans(K, n_init=20, random_state=SEED).fit(scaler.transform(snap[STYLE_INPUTS]))
    # canonical, label-permutation-proof naming from the centroids
    cen = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=STYLE_INPUTS)
    left, name_of = list(cen.index), {}
    for name, col in [("Wrestler / controller", "td15"), ("Submission hunter", "sub15"),
                      ("Clinch grinder", "clinch_share"), ("Leg-kick striker", "leg_share")]:
        j = cen.loc[left, col].idxmax()
        name_of[j] = name
        left.remove(j)
    name_of[left[0]] = "Volume striker"
    order = {old: NAME_ORDER.index(nm) for old, nm in name_of.items()}      # stable ids 0..4
    return scaler, km, order, snap


def assign_styles(hist, scaler, km, order):
    s = np.full(len(hist), UNKNOWN)
    ok = (hist.n_stat_fights >= MIN_STAT_FIGHTS) & hist[STYLE_INPUTS].notna().all(axis=1)
    raw = km.predict(scaler.transform(hist.loc[ok, STYLE_INPUTS]))
    s[ok.values] = [order[r] for r in raw]
    return s


def edge_matrix(fights, mask_train):
    """sum / count of (a_wins - Elo expectation) by ordered (self style, opponent style), antisymmetric."""
    f = fights[mask_train & (fights.A_style >= 0) & (fights.B_style >= 0)]
    S, N = np.zeros((K, K)), np.zeros((K, K))
    for a, b, r in zip(f.A_style, f.B_style, f.resid):
        S[a, b] += r; N[a, b] += 1
        S[b, a] -= r; N[b, a] += 1
    return S, N


def add_style_features(fights, hist):
    scaler, km, order, snap = fit_style_model(hist)
    hist = hist.copy()
    hist["style_id"] = assign_styles(hist, scaler, km, order)
    hist["style_name"] = hist.style_id.map(dict(enumerate(NAME_ORDER))).fillna("Unknown")
    key = hist.set_index(["fight_id", "fighter_id"]).style_id
    f = fights.copy()
    f["A_style"] = key.reindex(list(zip(f.fight_id, f.a_id))).values
    f["B_style"] = key.reindex(list(zip(f.fight_id, f.b_id))).values
    f["p_elo"] = 1 / (1 + 10 ** (-f.diff_elo / 400))
    f["resid"] = f.a_wins - f.p_elo

    tr = (f.meta_split == "train").values
    S, N = edge_matrix(f, tr)
    both = (f.A_style >= 0) & (f.B_style >= 0)
    a, b = f.A_style.clip(lower=0).values, f.B_style.clip(lower=0).values
    edge = S[a, b] / (N[a, b] + SHRINK)
    loo = (S[a, b] - f.resid.values) / (N[a, b] - 1 + SHRINK)          # leave-one-out for train rows
    edge = np.where(tr, loo, edge)
    same = a == b
    f["diff_style_edge"] = np.where(both & ~same, edge, 0.0)
    for k in range(-1, K):
        f[f"diff_style_{'unk' if k < 0 else k}"] = (f.A_style == k).astype(int) - (f.B_style == k).astype(int)
    f["style_known_both"] = both.astype(int)
    return f, hist, scaler, km, order, snap, (S, N)


def profiles(snap, scaler, km, order):
    X = snap[STYLE_INPUTS].copy()
    X["style_id"] = [order[r] for r in km.predict(scaler.transform(snap[STYLE_INPUTS]))]
    p = X.groupby("style_id")[STYLE_INPUTS].mean()
    p.insert(0, "style", [NAME_ORDER[i] for i in p.index])
    p.insert(1, "n_fighters", X.groupby("style_id").size())
    return p


def make_figures(prof, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    z = (prof[STYLE_INPUTS] - prof[STYLE_INPUTS].mean()) / prof[STYLE_INPUTS].std(ddof=0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(STYLE_INPUTS)), STYLE_INPUTS, rotation=35, ha="right")
    ax.set_yticks(range(len(prof)), [f"{s} (n={n})" for s, n in zip(prof["style"], prof["n_fighters"])])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{prof[STYLE_INPUTS].values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("What each style looks like (cell = average value, colour = relative to other styles)")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout(); fig.savefig(FIG / "style_profiles.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"wspace": 0.75})
    for ax, (title, sel) in zip(axes, [("Train period 2014-2022", f.meta_split == "train"),
                                       ("Test period 2024-2026 (out of sample)", f.meta_split == "test")]):
        g = f[sel & (f.A_style >= 0) & (f.B_style >= 0)]
        M, C = np.full((K, K), np.nan), np.zeros((K, K), int)
        for s in range(K):
            for t in range(K):
                if s == t:
                    continue
                x = np.r_[g.resid[(g.A_style == s) & (g.B_style == t)], -g.resid[(g.A_style == t) & (g.B_style == s)]]
                C[s, t] = len(x)
                if len(x) >= 15:
                    M[s, t] = 100 * x.mean()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-20, vmax=20)
        for s in range(K):
            for t in range(K):
                if s == t:
                    ax.text(t, s, "-", ha="center", va="center", color="grey")
                elif np.isnan(M[s, t]):
                    ax.text(t, s, f"n={C[s, t]}", ha="center", va="center", fontsize=7, color="grey")
                else:
                    ax.text(t, s, f"{M[s, t]:+.1f}\nn={C[s, t]}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(K), NAME_ORDER, rotation=35, ha="right"); ax.set_yticks(range(K), NAME_ORDER)
        ax.set_xlabel("Opponent style"); ax.set_ylabel("Fighter style"); ax.set_title(title)
    fig.suptitle("Style edge: win-rate points above/below what Elo predicts (row style vs column style)")
    fig.colorbar(im, ax=axes, shrink=0.7, label="percentage points"); fig.savefig(FIG / "style_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    hist = pd.read_csv(PROC / "fighter_history.csv", parse_dates=["date"])
    fights = pd.read_csv(PROC / "fight_features.csv", parse_dates=["meta_date"])
    f, hist2, scaler, km, order, snap, _ = add_style_features(fights, hist)
    prof = profiles(snap, scaler, km, order)
    hist2[["fight_id", "fighter_id", "date", "style_id", "style_name"]].to_csv(PROC / "fighter_styles.csv", index=False)
    prof.to_csv(PROC / "style_profiles.csv", index=False)
    f.to_csv(PROC / "fight_features_style.csv", index=False)
    make_figures(prof, f)
    print(prof.round(2).to_string(index=False))
    print(f"\nStyle known for both fighters: {f.style_known_both.mean():.1%} of modelling fights")
    print(f.groupby('meta_split').style_known_both.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
