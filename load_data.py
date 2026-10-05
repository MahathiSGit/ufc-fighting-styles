from pathlib import Path
from datasets import load_dataset

Path("data/raw").mkdir(parents=True, exist_ok=True)

rounds = load_dataset("JesterLabs/UFC_FIGHT_ROUNDS_DATA", split="train").to_pandas()
fights = load_dataset("JesterLabs/UFC_FIGHT_DATA", split="train").to_pandas()

rounds.to_csv("data/raw/rounds.csv", index=False)
fights.to_csv("data/raw/fights.csv", index=False)

print("rounds:", rounds.shape, "| fights:", fights.shape)
print("\nROUNDS columns:\n", list(rounds.columns))
print("\nFIGHTS columns:\n", list(fights.columns))

# Do the IDs match? This should be close to 1.0
match = rounds["FIGHT_ID"].isin(fights["FightId"]).mean()
print("\nrounds -> fights ID match:", round(match, 3))
print("fights with round data:", round(fights["FightId"].isin(rounds["FIGHT_ID"]).mean(), 3))
print("date range:", fights["Date"].min(), "->", fights["Date"].max())