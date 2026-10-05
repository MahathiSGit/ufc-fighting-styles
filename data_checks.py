"""Data quality checks for the UFC project. Read-only: it never changes any file.

Run after load_data.py and build_dataset.py:
    python data_checks.py

Checks: duplicates, winner names, unlinked fights, extreme values, early-UFC data,
and the 5-minute-round assumption.
"""
import argparse
from pathlib import Path

import pandas as pd


def mmss(x):
    try:
        m, s = str(x).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return None


def section(title):
    print(f"\n=== {title} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    a = ap.parse_args()
    raw, proc = Path(a.raw), Path(a.processed)

    # ---------------------------------------------------------------- raw tables
    fights_path, rounds_path = raw / "fights.csv", raw / "rounds.csv"
    if fights_path.exists():
        fights = pd.read_csv(fights_path, parse_dates=["Date"])

        section("1. Duplicates in the fights table")
        print("duplicate FightId values:", int(fights["FightId"].duplicated().sum()))
        print("fully duplicated rows   :", int(fights.duplicated().sum()))

        section("2. Winner names (should be Fighter 0, Fighter 1, or empty)")
        valid = (fights["Winner"] == fights["Fighter 0"]) | (fights["Winner"] == fights["Fighter 1"])
        empty = fights["Winner"].isna()
        print("winner matches a fighter:", int(valid.sum()))
        print("winner empty (draw / no contest):", int(empty.sum()))
        print("winner matches neither  :", int((~valid & ~empty).sum()), " <- should be 0")
        print("Fighter 0 is the winner in", f"{(fights['Winner'] == fights['Fighter 0'])[valid].mean():.1%}", "of decided fights")

        section("3. Round-length assumption (we assume 5-minute rounds)")
        secs = fights["Time"].map(mmss)
        print("longest time recorded in a round:", f"{secs.max() / 60:.1f} min", " <- should be <= 5.0")
        print("fights with unreadable Time     :", int(secs.isna().sum()))
        print("scheduled rounds (Max Round) counts:", fights["Max Round"].value_counts().to_dict())

        section("4. Early UFC (different rules and record keeping)")
        by_year = fights.groupby(fights["Date"].dt.year).size()
        print("fights per year before 2005:", by_year[by_year.index < 2005].to_dict())
        print("fights before 2001:", int((fights["Date"].dt.year < 2001).sum()),
              "of", len(fights))
    else:
        print(f"(skipping fights checks: {fights_path} not found)")
        fights = None

    if rounds_path.exists():
        rounds = pd.read_csv(rounds_path)
        section("5. Duplicates in the rounds table")
        key = ["FIGHT_ID", "FIGHTER_ID", "ROUND"]
        print("duplicate (fight, fighter, round) rows:", int(rounds.duplicated(key).sum()))
    else:
        print(f"(skipping rounds checks: {rounds_path} not found)")

    # ------------------------------------------------------- processed tables
    ff_path, wide_path = proc / "fighter_fights.csv", proc / "fights_wide.csv"
    if ff_path.exists():
        ff = pd.read_csv(ff_path, parse_dates=["Date"])

        section("6. Duplicates in the processed files")
        print("duplicate (fight, fighter) rows in fighter_fights:",
              int(ff.duplicated(["FightId", "FIGHTER_ID"]).sum()))
        if wide_path.exists():
            wide = pd.read_csv(wide_path, parse_dates=["Date"])
            print("duplicate FightId in fights_wide:", int(wide["FightId"].duplicated().sum()))
        print("fights with exactly 2 fighter rows:",
              f"{(ff.groupby('FightId').size() == 2).mean():.1%}", " <- should be 100%")

        section("7. Fights with no round data (style stats missing)")
        no_style = ff["sig_att_pm"].isna()
        print(f"fighter-fight rows without style stats: {int(no_style.sum()):,} ({no_style.mean():.1%})")
        print("by decade:", ff[no_style].groupby(ff["Date"].dt.year // 10 * 10).size().to_dict())

        section("8. Extreme values (per-fight style features)")
        cols = ["sig_att_pm", "absorbed_pm", "td_att_15", "sub_att_15", "kd_15"]
        print(ff[cols].describe(percentiles=[.5, .99, .999]).round(2).to_string())
        print("\nrows above the 99.9th percentile (candidates for capping):")
        for c in cols:
            thr = ff[c].quantile(0.999)
            print(f"  {c:<12} > {thr:7.2f}: {int((ff[c] > thr).sum())} rows, max = {ff[c].max():.1f}")
        print("\nmost extreme sig_att_pm rows:")
        top = ff.nlargest(5, "sig_att_pm")[["Date", "name", "sig_att_pm", "td_att_15", "kd_15"]]
        print(top.round(1).to_string(index=False))
        print("\nshare of values outside [0, 1] for rate-type columns (should be 0):")
        share_cols = ["sig_acc", "head_share", "body_share", "leg_share", "distance_share",
                      "clinch_share", "ground_share", "td_acc", "td_def", "ctrl_share"]
        bad = {c: int(((ff[c] < 0) | (ff[c] > 1)).sum()) for c in share_cols}
        print(bad)
    else:
        print(f"(skipping processed checks: {ff_path} not found; run build_dataset.py first)")


if __name__ == "__main__":
    main()