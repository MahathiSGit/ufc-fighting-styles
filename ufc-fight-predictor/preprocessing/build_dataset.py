"""
Step 2 - build the modelling dataset from the Kaggle UFC tables.

Source : data/raw/master.csv (one row per fight, fight-level totals + fighter bios)
         (Kaggle: neelagiriaditya/ufc-datasets-1994-2025)
Outputs: data/processed/fighter_history.csv      one row per fighter per fight, PRE-fight features
         data/processed/fight_features.csv       one row per fight: A_/B_/diff_ features + target a_wins
         data/processed/fighter_latest_state.csv each fighter's state after their latest fight (for the demo)

Leakage rules (enforced + tested in tests/test_no_leakage.py):
  * Every feature for a fight uses ONLY fights on strictly earlier dates.
  * The career-stat columns shipped in the Kaggle file (slpm, str_acc, ...) are snapshots taken
    at scrape time, so they are NEVER used. Everything is rebuilt from past fights.
  * Red/Blue corner is not a feature. Before 2010 the red corner is the winner 100% of the time
    (listing-order artifact), so fighter A/B is assigned at random (fixed seed).
  * Shrinkage priors come only from fights before PRIOR_END (before the modelling window).
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"

MODEL_START = pd.Timestamp("2014-01-01")   # first fight that becomes a modelling row
PRIOR_END = pd.Timestamp("2014-01-01")     # priors use fights before this date only
VAL_START, TEST_START = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")
SEED = 42

# short name -> master.csv column suffix (own = r_/b_ prefix)
STAT_BASES = {
    "kd": "total_kd", "sig_l": "total_sig_landed", "sig_a": "total_sig_atmp",
    "td_l": "total_td_success", "td_a": "total_td_atmp", "sub": "total_sub_att", "rev": "total_rev",
    "head_l": "total_sig_str_landed_head", "body_l": "total_sig_str_landed_body",
    "leg_l": "total_sig_str_landed_leg", "dist_l": "total_sig_str_landed_distance",
    "clinch_l": "total_sig_str_landed_clinch", "ground_l": "total_sig_str_landed_ground",
    "ctrl": "total_ctrl_seconds",
}
OWN = list(STAT_BASES)
OPP_USED = ["kd", "sig_l", "sig_a", "td_l", "td_a", "ctrl"]
COUNTS = ["n", "n_stat", "wins", "losses", "draws", "ko_w", "sub_w", "dec_w", "ko_l", "sub_l", "secs"]
SUM_COLS = COUNTS + OWN + ["opp_" + k for k in OPP_USED]

STYLE_COLS = ["slpm", "head_share", "body_share", "leg_share", "dist_share", "clinch_share",
              "ground_share", "td15", "sub15", "ctrl_share"]
DERIVED = ["n_fights", "n_stat_fights", "win_rate", "ko_w_rate", "sub_w_rate", "dec_w_rate",
           "ko_l_rate", "sub_l_rate", "slpm", "sapm", "sig_diff_pm", "sig_acc", "sig_def",
           "td15", "td_acc", "td_def", "sub15", "ctrl_share", "kd15", "kda15",
           "head_share", "body_share", "leg_share", "dist_share", "clinch_share", "ground_share"]
EXTRA = ["age", "height_in", "reach_in", "elo", "streak", "layoff_days"]
NUM_FEATS = DERIVED + EXTRA


# ------------------------------------------------------------------ loading
def parse_height(s):
    mm = re.match(r"\s*(\d+)'\s*(\d+)", s) if isinstance(s, str) else None
    return int(mm.group(1)) * 12 + int(mm.group(2)) if mm else np.nan


def fight_seconds(time_format, finish_round, finish_time):
    """Elapsed fight time in seconds using the round lengths in time_format, e.g. '3 Rnd (5-5-5)'."""
    lens = [int(x) for x in re.findall(r"\d+", (re.search(r"\(([\d\-]+)\)", str(time_format)) or [None, ""])[1])]
    mm = re.match(r"(\d+):(\d+)", str(finish_time))
    if not mm or pd.isna(finish_round):
        return np.nan
    r = int(finish_round)
    before = sum(lens[: r - 1]) * 60 if lens else (r - 1) * 300
    return before + int(mm.group(1)) * 60 + int(mm.group(2))


def load_master(path=None):
    m = pd.read_csv(path or RAW / "master.csv", low_memory=False, encoding="utf-8-sig")
    m["date"] = pd.to_datetime(m["event_date"])
    m = m[m.event_name.str.contains(r"UFC|Ultimate Fighter", case=False, na=False)].copy()
    m["has_stats"] = m.rounds_fought.fillna(0) > 0           # master shows zeros when round data is missing
    m["duration"] = [fight_seconds(a, b, c) for a, b, c in zip(m.time_format, m.finish_round, m.finish_time)]
    m["num_rounds"] = m.time_format.str.extract(r"^(\d+)")[0].astype(float)
    return m.sort_values(["date", "fight_id"]).reset_index(drop=True)


def to_long(m):
    """Two rows per fight (one per fighter) with own and opponent totals."""
    ok = m.has_stats & m.duration.notna()
    parts = []
    for me, op in (("r", "b"), ("b", "r")):
        d = pd.DataFrame({
            "fight_id": m.fight_id, "date": m.date, "fighter_id": m[f"{me}_fighter_id"],
            "opp_id": m[f"{op}_fighter_id"], "name": m[f"{me}_fighter_name"],
            "dob": pd.to_datetime(m[f"{me}_dob"], errors="coerce"),
            "height_in": m[f"{me}_height"].map(parse_height), "reach_in": m[f"{me}_reach_inches"],
            "stance": m[f"{me}_stance"], "is_red": me == "r",
        })
        for k, b in STAT_BASES.items():
            d[k] = m[f"{me}_{b}"].where(ok, 0).fillna(0)
        for k in OPP_USED:
            d["opp_" + k] = m[f"{op}_{STAT_BASES[k]}"].where(ok, 0).fillna(0)
        res = np.where(m.result_status == "win", np.where(m.winner_id == d.fighter_id, "win", "loss"),
                       np.where(m.result_status == "draw", "draw", "nc"))
        d["result"] = res
        meth = m.method.fillna("")
        ko, sb, dc = meth.str.contains("KO"), meth.str.contains("Submission"), meth.str.startswith("Decision")
        w, l = res == "win", res == "loss"
        d["n"], d["n_stat"] = 1, ok.astype(int)
        d["wins"], d["losses"], d["draws"] = w.astype(int), l.astype(int), (res == "draw").astype(int)
        d["ko_w"], d["sub_w"], d["dec_w"] = (w & ko).astype(int), (w & sb).astype(int), (w & dc).astype(int)
        d["ko_l"], d["sub_l"] = (l & ko).astype(int), (l & sb).astype(int)
        d["secs"] = m.duration.where(ok, 0).fillna(0)
        parts.append(d)
    L = pd.concat(parts, ignore_index=True)
    return L.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ features
def compute_priors(L):
    s = L.loc[L.date < PRIOR_END, SUM_COLS].sum()
    mins = s.secs / 60
    return {
        "slpm": s.sig_l / mins, "sapm": s.opp_sig_l / mins, "acc": s.sig_l / s.sig_a,
        "td_pm": s.td_l / mins, "td_acc": s.td_l / s.td_a, "sub_pm": s["sub"] / mins,
        "ctrl_share": s.ctrl / s.secs, "kd_pm": s.kd / mins, "kda_pm": s.opp_kd / mins,
        "head": s.head_l / s.sig_l, "body": s.body_l / s.sig_l, "leg": s.leg_l / s.sig_l,
        "dist": s.dist_l / s.sig_l, "clinch": s.clinch_l / s.sig_l, "ground": s.ground_l / s.sig_l,
        **{k + "_rate": s[k] / s.n for k in ["ko_w", "sub_w", "dec_w", "ko_l", "sub_l"]},
    }


def derive(c, P):
    """Cumulative sums (from PAST fights) -> smoothed rate features. Shrinks toward the prior."""
    K, A, S = 15.0, 100.0, 50.0           # pseudo-minutes, pseudo-attempts, pseudo-strikes
    mins = c.secs / 60
    f = pd.DataFrame(index=c.index)
    f["n_fights"], f["n_stat_fights"] = c.n, c.n_stat
    f["win_rate"] = (c.wins + 0.5 * 4) / (c.n + 4)
    for k in ["ko_w", "sub_w", "dec_w", "ko_l", "sub_l"]:
        f[k + "_rate"] = (c[k] + P[k + "_rate"] * 3) / (c.n + 3)
    f["slpm"] = (c.sig_l + P["slpm"] * K) / (mins + K)
    f["sapm"] = (c.opp_sig_l + P["sapm"] * K) / (mins + K)
    f["sig_diff_pm"] = f.slpm - f.sapm
    f["sig_acc"] = (c.sig_l + P["acc"] * A) / (c.sig_a + A)
    f["sig_def"] = 1 - (c.opp_sig_l + P["acc"] * A) / (c.opp_sig_a + A)
    f["td15"] = 15 * (c.td_l + P["td_pm"] * K) / (mins + K)
    f["td_acc"] = (c.td_l + P["td_acc"] * 10) / (c.td_a + 10)
    f["td_def"] = 1 - (c.opp_td_l + P["td_acc"] * 10) / (c.opp_td_a + 10)
    f["sub15"] = 15 * (c["sub"] + P["sub_pm"] * K) / (mins + K)
    f["ctrl_share"] = (c.ctrl + P["ctrl_share"] * K * 60) / (c.secs + K * 60)
    f["kd15"] = 15 * (c.kd + P["kd_pm"] * K) / (mins + K)
    f["kda15"] = 15 * (c.opp_kd + P["kda_pm"] * K) / (mins + K)
    for out, col, p in [("head_share", "head_l", "head"), ("body_share", "body_l", "body"),
                        ("leg_share", "leg_l", "leg"), ("dist_share", "dist_l", "dist"),
                        ("clinch_share", "clinch_l", "clinch"), ("ground_share", "ground_l", "ground")]:
        f[out] = (c[col] + P[p] * S) / (c.sig_l + S)
    return f


def streak_and_layoff(L):
    """Current win(+)/loss(-) streak and days since last fight, using strictly earlier dates only."""
    n = len(L)
    pre_streak, layoff = np.zeros(n), np.full(n, np.nan)
    fid, dates, res = L.fighter_id.values, L.date.values, L.result.values
    final, i = {}, 0
    while i < n:
        j = i
        while j < n and fid[j] == fid[i]:
            j += 1
        streak, last = 0, None
        k = i
        while k < j:
            e = k
            while e < j and dates[e] == dates[k]:
                e += 1
            for t in range(k, e):                       # same-date rows see the same committed state
                pre_streak[t] = streak
                if last is not None:
                    layoff[t] = (dates[k] - last) / np.timedelta64(1, "D")
            for t in range(k, e):                       # commit after the date is done
                if res[t] == "win":
                    streak = streak + 1 if streak > 0 else 1
                elif res[t] == "loss":
                    streak = streak - 1 if streak < 0 else -1
                elif res[t] == "draw":
                    streak = 0
            last = dates[k]
            k = e
        final[fid[i]] = streak
        i = j
    return pre_streak, layoff, final


def elo_ratings(L, k_factor=32.0):
    """Elo before each fight (all UFC fights, draws = 0.5, no-contests ignored)."""
    pre = {}
    R = defaultdict(lambda: 1500.0)
    fights = L[L.is_red].merge(L[~L.is_red][["fight_id", "fighter_id", "result"]], on="fight_id",
                               suffixes=("", "_b")).sort_values(["date", "fight_id"])
    for _, g in fights.groupby("date", sort=True):
        for r in g.itertuples():
            pre[(r.fight_id, r.fighter_id)], pre[(r.fight_id, r.fighter_id_b)] = R[r.fighter_id], R[r.fighter_id_b]
        for r in g.itertuples():
            if r.result == "nc":
                continue
            s_r = {"win": 1.0, "loss": 0.0, "draw": 0.5}[r.result]
            e_r = 1 / (1 + 10 ** ((R[r.fighter_id_b] - R[r.fighter_id]) / 400))
            R[r.fighter_id] += k_factor * (s_r - e_r)
            R[r.fighter_id_b] += k_factor * ((1 - s_r) - (1 - e_r))
    return pre, dict(R)


def build_all(m):
    L = to_long(m)
    P = compute_priors(L)
    cum = L.groupby("fighter_id")[SUM_COLS].cumsum()
    keys = L[["fighter_id", "date"]]
    pre = (cum - L[SUM_COLS]).join(keys).groupby(["fighter_id", "date"])[SUM_COLS].transform("min")

    hist = L[["fight_id", "fighter_id", "date", "name", "dob", "height_in", "reach_in", "stance", "is_red"]].copy()
    hist = hist.join(derive(pre, P))
    streak, layoff, streak_final = streak_and_layoff(L)
    elo_pre, elo_final = elo_ratings(L)
    hist["streak"], hist["layoff_days"] = streak, layoff
    hist["elo"] = [elo_pre[(a, b)] for a, b in zip(L.fight_id, L.fighter_id)]
    hist["age"] = (hist.date - hist.dob).dt.days / 365.25
    hist["reach_missing"] = hist.reach_in.isna().astype(int)

    last = cum.groupby(L.fighter_id).tail(1)
    latest = L.loc[last.index, ["fighter_id", "name", "dob", "height_in", "reach_in", "stance", "date"]].rename(
        columns={"date": "last_fight_date"})
    latest = latest.join(derive(last, P))
    latest["elo"] = latest.fighter_id.map(elo_final)
    latest["streak"] = latest.fighter_id.map(streak_final)
    latest["reach_missing"] = latest.reach_in.isna().astype(int)

    fights = build_fights(m, hist)
    return L, hist, fights, latest.reset_index(drop=True), P


def build_fights(m, hist):
    f = m[(m.result_status == "win") & (m.date >= MODEL_START)].sort_values(["date", "fight_id"]).copy()
    f["a_is_red"] = np.random.default_rng(SEED).random(len(f)) < 0.5     # random corner -> no ordering leak
    f["a_id"] = np.where(f.a_is_red, f.r_fighter_id, f.b_fighter_id)
    f["b_id"] = np.where(f.a_is_red, f.b_fighter_id, f.r_fighter_id)
    out = pd.DataFrame({
        "fight_id": f.fight_id.values, "meta_date": f.date.values, "meta_event": f.event_name.values,
        "meta_weight_class": f.weight_class.values, "meta_title_fight": f.title_fight.values,
        "meta_num_rounds": f.num_rounds.values, "meta_method": f.method.values,
        "meta_a_is_red": f.a_is_red.values, "a_id": f.a_id.values, "b_id": f.b_id.values,
        "a_wins": (f.winner_id == f.a_id).astype(int).values,
    })
    cols = NUM_FEATS + ["reach_missing"]
    for side in ("a", "b"):
        h = hist[["fight_id", "fighter_id", "name", "stance"] + cols].rename(
            columns={c: f"{side.upper()}_{c}" for c in cols + ["name", "stance"]})
        out = out.merge(h, left_on=["fight_id", f"{side}_id"], right_on=["fight_id", "fighter_id"], how="left"
                        ).drop(columns="fighter_id")
    for c in NUM_FEATS:
        out[f"diff_{c}"] = out[f"A_{c}"] - out[f"B_{c}"]
    out["meta_split"] = np.where(out.meta_date >= TEST_START, "test",
                                 np.where(out.meta_date >= VAL_START, "val", "train"))
    return out


def feature_columns(fights):
    """Columns a model may use. Anything starting with meta_ / id / target is excluded by construction."""
    return [c for c in fights.columns if c.startswith(("A_", "B_", "diff_")) and not c.endswith(("_name",))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_master()
    L, hist, fights, latest, P = build_all(m)
    hist.to_csv(OUT / "fighter_history.csv", index=False)
    fights.to_csv(OUT / "fight_features.csv", index=False)
    latest.to_csv(OUT / "fighter_latest_state.csv", index=False)
    print(f"UFC fights loaded: {len(m)} | fighter-fight rows: {len(L)} | fighters: {L.fighter_id.nunique()}")
    print(f"Modelling rows: {len(fights)} | target mean (a_wins): {fights.a_wins.mean():.3f}")
    print(fights.meta_split.value_counts().to_string())
    print("Wrote", [p.name for p in OUT.glob('*.csv')])


if __name__ == "__main__":
    main()
