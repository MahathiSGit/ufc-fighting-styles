"""Build pre-fight style profiles from the two Hugging Face tables.

Input  (from load_data.py):  data/raw/rounds.csv, data/raw/fights.csv
Output (data/processed/):
    fighter_fights.csv  one row per fighter per fight, with the fighter's style
                        profile computed ONLY from their previous fights
    fights_wide.csv     one row per fight: fighter 0 vs fighter 1 profiles + label

Notes on fight length (validated with data_checks.py):
    minutes = (final_round - 1) * 5 + time_in_final_round.
    This assumes 5-minute rounds. In the data, the only rounds longer than 5 minutes
    are 48 early fights (1994-1999); 42 of them are single-round fights (Max Round = 1),
    where the formula simply uses the recorded time, so it is exact. The other 6 have
    no Max Round value. No correction is needed.

Usage:  python build_dataset.py [--window 5] [--min-prior 2]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---- columns used (names from rounds.csv) ---------------------------------
SUM_COLS = [
    "HEAD LANDED", "HEAD ATTEMPTED", "BODY LANDED", "BODY ATTEMPTED",
    "LEG LANDED", "LEG ATTEMPTED", "DISTANCE LANDED", "DISTANCE ATTEMPTED",
    "CLINCH LANDED", "CLINCH ATTEMPTED", "GROUND LANDED", "GROUND ATTEMPTED",
    "KD", "SUB ATTEMPTS", "REVERSALS", "TD LANDED", "TD ATTEMPTED", "CTRL SEC",
]

# Style features (computed per fight, then averaged over past fights)
STYLE_COLS = [
    "sig_att_pm",      # significant strikes attempted per minute (volume)
    "sig_acc",         # strike accuracy
    "absorbed_pm",     # significant strikes absorbed per minute
    "head_share", "body_share", "leg_share",
    "distance_share", "clinch_share", "ground_share",
    "td_att_15", "td_acc", "td_def",
    "ctrl_share",      # fraction of fight time spent in control
    "sub_att_15", "kd_15",
]


def mmss_to_sec(x):
    """'2:35' -> 155. '--', blanks, NaN -> NaN."""
    if pd.isna(x):
        return np.nan
    m = re.match(r"^\s*(\d+):(\d{1,2})\s*$", str(x))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan


def safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--window", type=int, default=5, help="past fights to average")
    ap.add_argument("--min-prior", type=int, default=2, help="min past fights needed")
    a = ap.parse_args()

    rounds = pd.read_csv(Path(a.raw) / "rounds.csv")
    fights = pd.read_csv(Path(a.raw) / "fights.csv", parse_dates=["Date"])

    # ---- 1. clean round data -------------------------------------------------
    rounds["CTRL SEC"] = rounds["CTRL TIME"].map(mmss_to_sec)
    for c in SUM_COLS:
        rounds[c] = pd.to_numeric(rounds[c], errors="coerce")
    agg = (rounds.groupby(["FIGHT_ID", "FIGHTER_ID"], as_index=False)[SUM_COLS]
                 .sum(min_count=1))

    # ---- 1b. re-link rounds to fights by FIGHTER PAIR -------------------------
    # The FIGHT_ID in the rounds table does not point to the same fight as FightId
    # in the fights table, but the two fighters inside each round-table fight are
    # real opponents. So we match on the unordered pair of fighter IDs; for
    # rematches (same pair, several fights) we use the number of rounds.
    rf = (rounds.groupby("FIGHT_ID")
                .agg(n_fighters=("FIGHTER_ID", "nunique"),
                     n_rounds=("ROUND", "max"),
                     pair=("FIGHTER_ID", lambda s: "|".join(sorted(set(s)))))
                .reset_index())
    rf = rf[rf.n_fighters == 2]
    fights["pair"] = ["|".join(sorted([x, y])) for x, y in
                      zip(fights["Fighter 0 Id"], fights["Fighter 1 Id"])]
    cand = fights[["FightId", "pair", "Round"]].merge(
        rf[["FIGHT_ID", "pair", "n_rounds"]], on="pair")
    uniq = ((cand.groupby("FightId")["FIGHT_ID"].transform("nunique") == 1) &
            (cand.groupby("FIGHT_ID")["FightId"].transform("nunique") == 1))
    amb = cand[~uniq & (cand["Round"] == cand["n_rounds"])]
    amb = amb[(amb.groupby("FightId")["FIGHT_ID"].transform("nunique") == 1) &
              (amb.groupby("FIGHT_ID")["FightId"].transform("nunique") == 1)]
    mapping = pd.concat([cand[uniq], amb]).drop_duplicates(["FightId", "FIGHT_ID"])
    mapping = mapping.drop_duplicates("FightId", keep=False).drop_duplicates("FIGHT_ID", keep=False)
    print(f"Re-linked {len(mapping):,} of {len(fights):,} fights to round data "
          f"({len(mapping)/len(fights):.1%}); rematches resolved by round count: {len(amb):,}")
    chk = (mapping["Round"] == mapping["n_rounds"]).mean()
    print(f"Sanity check - last round in fights table == rounds in round table: {chk:.1%}")
    agg = agg.merge(mapping[["FIGHT_ID", "FightId"]], on="FIGHT_ID", how="inner").drop(columns="FIGHT_ID")

    # ---- 2. fight length in minutes (from the fights table) ------------------
    fights["end_sec"] = fights["Time"].map(mmss_to_sec)
    # (final round - 1) * 5 minutes of earlier rounds + time in the final round.
    # Exact for single-round fights (earlier-round term is 0), which are the only
    # fights whose round exceeds 5 minutes. Floor of 15 seconds avoids divide-by-tiny.
    fights["minutes"] = ((fights["Round"] - 1) * 5 + fights["end_sec"] / 60).clip(lower=0.25)
    long_multi = (fights["end_sec"] > 300) & (fights["Round"] > 1)
    if long_multi.any():
        print(f"WARNING: {int(long_multi.sum())} multi-round fights have a round over 5 min; "
              "their length may be slightly off")

    # ---- 3. long table: one row per fighter per fight ------------------------
    f0 = fights.rename(columns={"Fighter 0 Id": "FIGHTER_ID", "Fighter 0": "name",
                                "Fighter 1 Id": "OPP_ID"})
    f1 = fights.rename(columns={"Fighter 1 Id": "FIGHTER_ID", "Fighter 1": "name",
                                "Fighter 0 Id": "OPP_ID"})
    keep = ["FightId", "Date", "FIGHTER_ID", "OPP_ID", "name", "Winner", "Method",
            "Weight class", "minutes"]
    long = pd.concat([f0[keep].assign(slot=0), f1[keep].assign(slot=1)], ignore_index=True)
    long["won"] = (long["Winner"] == long["name"]).astype(int)
    long["finished"] = long["Method"].isin(["KO/TKO", "SUB"]).astype(int)

    long = long.merge(agg, on=["FightId", "FIGHTER_ID"], how="left")
    opp = agg.rename(columns={"FIGHTER_ID": "OPP_ID", **{c: f"opp_{c}" for c in SUM_COLS}})
    long = long.merge(opp, on=["FightId", "OPP_ID"], how="left")

    # ---- 4. per-fight style features ----------------------------------------
    m = long["minutes"]
    sig_att = long[["HEAD ATTEMPTED", "BODY ATTEMPTED", "LEG ATTEMPTED"]].sum(axis=1, min_count=1)
    sig_land = long[["HEAD LANDED", "BODY LANDED", "LEG LANDED"]].sum(axis=1, min_count=1)
    opp_sig_land = long[["opp_HEAD LANDED", "opp_BODY LANDED", "opp_LEG LANDED"]].sum(axis=1, min_count=1)

    long["sig_att_pm"] = sig_att / m
    long["sig_acc"] = safe_div(sig_land, sig_att)
    long["absorbed_pm"] = opp_sig_land / m
    for t in ["HEAD", "BODY", "LEG"]:
        long[f"{t.lower()}_share"] = safe_div(long[f"{t} ATTEMPTED"], sig_att)
    pos_att = long[["DISTANCE ATTEMPTED", "CLINCH ATTEMPTED", "GROUND ATTEMPTED"]].sum(axis=1, min_count=1)
    for t in ["DISTANCE", "CLINCH", "GROUND"]:
        long[f"{t.lower()}_share"] = safe_div(long[f"{t} ATTEMPTED"], pos_att)
    long["td_att_15"] = long["TD ATTEMPTED"] / m * 15
    long["td_acc"] = safe_div(long["TD LANDED"], long["TD ATTEMPTED"])
    long["td_def"] = 1 - safe_div(long["opp_TD LANDED"], long["opp_TD ATTEMPTED"])
    long["ctrl_share"] = (long["CTRL SEC"] / (m * 60)).clip(upper=1)
    long["sub_att_15"] = long["SUB ATTEMPTS"] / m * 15
    long["kd_15"] = long["KD"] / m * 15

    # ---- 5. PRE-FIGHT profile: average of previous fights only ---------------
    long = long.sort_values(["Date", "FightId", "slot"]).reset_index(drop=True)
    g = long.groupby("FIGHTER_ID", sort=False)
    prior = g[STYLE_COLS].transform(
        lambda s: s.shift(1).rolling(a.window, min_periods=1).mean())
    prior.columns = [f"pre_{c}" for c in STYLE_COLS]
    long = pd.concat([long, prior], axis=1)
    long["n_prior"] = g.cumcount()                       # fights before this one
    long["pre_win_rate"] = g["won"].transform(lambda s: s.shift(1).expanding().mean())
    long["pre_finish_rate"] = g["finished"].transform(lambda s: s.shift(1).expanding().mean())
    long["has_history"] = long["n_prior"] >= a.min_prior

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cols = (["FightId", "Date", "FIGHTER_ID", "name", "OPP_ID", "slot", "won",
             "n_prior", "has_history", "pre_win_rate", "pre_finish_rate"]
            + [f"pre_{c}" for c in STYLE_COLS] + STYLE_COLS)
    long[cols].to_csv(out / "fighter_fights.csv", index=False)

    # ---- 6. wide table: one row per fight -----------------------------------
    pre = ["n_prior", "has_history", "pre_win_rate", "pre_finish_rate"] + [f"pre_{c}" for c in STYLE_COLS]
    w0 = long[long.slot == 0].set_index("FightId")[["Date", "FIGHTER_ID", "name", "won"] + pre].add_prefix("f0_")
    w1 = long[long.slot == 1].set_index("FightId")[["FIGHTER_ID", "name", "won"] + pre].add_prefix("f1_")
    wide = w0.join(w1, how="inner").reset_index()
    wide = wide.rename(columns={"f0_Date": "Date"})
    decisive = (wide["f0_won"] + wide["f1_won"]) == 1       # drop draws / no contests
    wide["label_f0_won"] = wide["f0_won"]
    wide = wide[decisive].sort_values("Date")
    wide.to_csv(out / "fights_wide.csv", index=False)

    usable = wide[(wide.f0_has_history) & (wide.f1_has_history)]
    print(f"fighter_fights.csv : {len(long):,} rows")
    print(f"fights_wide.csv    : {len(wide):,} decisive fights")
    print(f"  with >= {a.min_prior} prior fights for BOTH fighters: {len(usable):,}")
    print(f"  date range of usable fights: {usable['Date'].min().date()} -> {usable['Date'].max().date()}")
    print("\nMissing share in style features (profile rows with history):")
    print(long.loc[long.has_history, [f'pre_{c}' for c in STYLE_COLS]].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()