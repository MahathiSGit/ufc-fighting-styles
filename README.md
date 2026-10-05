# Do Fighting Styles Matter? A Style-Based Analysis of UFC Fights

**Course:** UE24CS352A – Machine Learning Mini-Project
**Team:** `<Mahathi S (SRN: PES2UG24AM084)>` and `<Name 2>`

**Topic:** Predicting UFC fight outcomes (inspired by *"Applying Machine Learning Algorithms to Predict UFC Fight Outcomes"*, McQuaide, Stanford)

---

## 1. The idea

Most UFC prediction projects look only at *skill* numbers: a fighter's record, age, reach, and so on. But mixed martial arts (MMA) is a little like rock-paper-scissors. A wrestler might dominate a striker, yet struggle against a submission specialist. A model that only asks *"who is the better fighter?"* can miss *"who is better against this kind of opponent?"*

This project tests that idea in three steps:

1. **Find fighting styles.** Describe how every fighter fights using their past fight statistics (how often they strike, where they aim, how often they take people down, how long they control opponents, and so on), then use clustering to group similar fighters into a small number of styles.
2. **Build a style-vs-style matchup chart.** For every pair of styles, measure how often one style beats the other.
3. **Test whether style helps prediction.** Predict fight winners with (a) skill features only, and (b) skill plus style features, and check whether style adds anything.

## 2. Goal 

> **Does knowing a fighter's style improve our ability to predict who wins a UFC fight, compared with skill-based information alone?**

"Yes", "a little", and "no" are all valid findings. The aim is to answer the question honestly, with proper safeguards against cheating (data leakage) and with uncertainty estimates.


---

## 3. The data

### 3.1 Where it comes from

All numbers originate from **UFCStats.com**, the official UFC statistics site. We could not scrape the site ourselves (it now shows a JavaScript "Checking your browser…" check that automated tools cannot pass), so we use two published datasets on Hugging Face that were derived from UFCStats:

| Dataset | Link | Used for |
|---|---|---|
| **Fights table** | https://huggingface.co/datasets/JesterLabs/UFC_FIGHT_DATA | date, fighters, winner, method, final round and time |
| **Rounds table** | https://huggingface.co/datasets/JesterLabs/UFC_FIGHT_ROUNDS_DATA | detailed round-by-round statistics (the raw material for "style") |

These two tables are the **only** data the project needs. Both can be browsed in the browser (open the dataset page and choose *Data Studio* / the viewer tab).

Coverage: **7,961 fights**, from **1994-03-11 to 2024-12-07**, involving **2,549 different fighters**. Because the datasets are a snapshot, nothing after December 2024 is included.

### 3.2 The fights table (`data/raw/fights.csv`, 7,961 rows)

**One row = one fight.** Main columns:

| Column | Meaning |
|---|---|
| `FightId` | UFCStats ID of the fight |
| `Date` | Date of the event |
| `Fighter 0`, `Fighter 1` | Names of the two fighters |
| `Fighter 0 Id`, `Fighter 1 Id` | UFCStats IDs of the fighters |
| `Winner` | Name of the winner (empty for draws / no contests) |
| `Outcome`, `Method`, `Move` | How the fight ended (e.g. `KO/TKO`, `SUB`, `U-DEC`) |
| `Round`, `Time` | Round and time in the round when the fight ended |
| `Max Round` | Scheduled number of rounds (3 or 5) |
| `Weight class`, `Title`, `Location`, `Url` | Context about the fight |
| `Fighter 0/1 Kd, Str, Td, Sub` | Whole-fight totals: knockdowns, strikes, takedowns, submission attempts |

> **Important:** in this table **Fighter 0 is always the winner**, because UFCStats lists the winner first. This means (a) the data contains **no red/blue corner information**, and (b) **fighter order must never be used as a feature**: a model that saw it would score 100% by cheating. Our code only uses *differences* between the two fighters and trains on both orderings, so it cannot exploit this.

### 3.3 The rounds table (`data/raw/rounds.csv`, 37,368 rows)

**One row = one fighter in one round of one fight.** It is much bigger than the fights table because each fight has two fighters and roughly 2–3 rounds (7,961 fights × 2 fighters × ~2.35 rounds ≈ 37,000 rows).

| Column group | Columns | Meaning |
|---|---|---|
| Identity | `FIGHTER`, `FIGHTER_ID`, `FIGHT_ID`, `FIGHTER_SLUG`, `FIGHTER_URL`, `ROUND` | Who, which fight, which round |
| Strikes by **target** | `HEAD`, `BODY`, `LEG` × `LANDED` / `ATTEMPTED` | Where the fighter aimed |
| Strikes by **position** | `DISTANCE`, `CLINCH`, `GROUND` × `LANDED` / `ATTEMPTED` | Where the strike happened: standing at range, in the clinch, or on the ground |
| Grappling | `TD LANDED`, `TD ATTEMPTED`, `SUB ATTEMPTS`, `REVERSALS`, `CTRL TIME` | Takedowns, submission attempts, reversals, time spent in control (as `m:ss` text) |
| Power | `KD` | Knockdowns |
| Other | `SIG STR…`, `TOTAL STR…`, `TD %` | Totals and percentages (some columns appear twice with `_x` / `_y` suffixes; we do **not** use them) |

### 3.4 A data-quality problem we found and fixed

The `FIGHT_ID` values in the rounds table **do not point to the same fights** as `FightId` in the fights table. Example: the same ID is *Julianna Pena vs Raquel Pennington* in the fights table but *Carla Esparza vs Tecia Pennington* in the rounds table. A naive join on the ID looked fine at first (every ID exists in both tables) but only about **9%** of fighter-fights matched correctly, leaving about 90% of the style features empty.

**Fix:** the *fighter* IDs are consistent between the tables, and each fight in the rounds table contains a genuine pair of opponents. So we link the tables by the **unordered pair of fighter IDs**. If the same two fighters met more than once (rematches), we separate the fights by number of rounds; fights that remain ambiguous are skipped rather than guessed.

Result: **7,791 of 7,961 fights (97.9%)** were linked to round data, and as a sanity check, the final round in the fights table equals the number of rounds in the rounds table for **100%** of linked fights.

### 3.5 What the processed files are

`build_dataset.py` turns the two raw tables into two processed files:

**`data/processed/fighter_fights.csv`**: **one row per fighter per fight** (15,922 rows). Used for clustering.

| Columns | Meaning |
|---|---|
| `FightId`, `Date`, `FIGHTER_ID`, `name`, `OPP_ID`, `slot` | Who fought whom and when (`slot` is the 0/1 position in the original table) |
| `won` | 1 if this fighter won, else 0 |
| `n_prior`, `has_history` | Number of earlier fights this fighter had; `has_history` = at least 2 |
| `pre_win_rate`, `pre_finish_rate` | Share of earlier fights won, and share ended by KO/TKO or submission |
| `pre_<style feature>` (15 columns) | The fighter's style **averaged over their previous 5 fights only** |
| `<style feature>` (15 columns, no `pre_`) | What the fighter actually did **in this fight** (used only to build later fights' `pre_` values) |

**`data/processed/fights_wide.csv`**: **one row per fight** (7,817 decisive fights; draws and no-contests removed). Used for the matchup chart and prediction. It holds `f0_…` columns (Fighter 0's profile before the fight), `f1_…` columns (Fighter 1's profile), and `label_f0_won`.

### 3.6 Why many cells are blank (and why that is fine)

About 10% of cells in `fighter_fights.csv` and 13% in `fights_wide.csv` are empty, and most *rows* have at least one gap. This is expected:

| Cause | Detail |
|---|---|
| **Debut fights** | `pre_` columns mean "average of earlier fights"; a fighter's first fight has none. 2,549 rows (16%) are debuts. We keep them because the debut still feeds the *next* fight's profile |
| **Control time in old events** | `ctrl_share` is blank for ~100% of 1990–94 fights and ~89% of 1995–99 fights, but only 1–4% from 2000 on (likely not recorded back then) |
| **Undefined ratios** | Takedown accuracy is blank if a fighter attempted no takedowns; takedown defense is blank if the opponent attempted none. Writing 0% would be wrong |
| **Unlinked fights** | ~2% of fights could not be linked to round data |

The files are sorted by date, so the top rows (1990s) look emptiest. For the actual analysis we only use fights where **both** fighters have at least 2 previous fights (**4,372 fights**); there, only ~3% of cells are blank, and those are filled with the median value from the training period.

---

## 4. Method

### 4.1 Style features (15 per fighter)

For each fight we compute these from the round data, then average them over the fighter's previous 5 fights.

| Group | Feature | Definition |
|---|---|---|
| Volume / quality | `sig_att_pm` | Significant strikes attempted per minute (head + body + leg) |
| | `sig_acc` | Strikes landed ÷ attempted |
| | `absorbed_pm` | Opponent's significant strikes landed per minute |
| Target | `head_share`, `body_share`, `leg_share` | Share of strikes aimed at each target |
| Position | `distance_share`, `clinch_share`, `ground_share` | Share of strikes thrown standing at range / in the clinch / on the ground |
| Grappling | `td_att_15` | Takedown attempts per 15 minutes |
| | `td_acc` | Takedowns landed ÷ attempted |
| | `td_def` | 1 − (opponent takedowns landed ÷ attempted) |
| | `ctrl_share` | Fraction of the fight spent in control |
| Finishing threat | `sub_att_15`, `kd_15` | Submission attempts and knockdowns per 15 minutes |

Fight length is computed from the final round and time, assuming 5-minute rounds.

### 4.2 Preventing data leakage ("What did we know *before* this fight?")

- Every style and record feature is computed **only from earlier fights** (`pre_` columns). A fighter's first fight has no profile.
- Clustering, the matchup chart, and the model are fitted on **training fights only**. The test set is touched once, at the end.
- **Time-based split:** train on fights **before 2020-01-01**, test on fights **from 2020 on** (2,768 train / 1,604 test fights). This mimics predicting the future.
- Within the training set, each fight's own result is **left out** when computing its matchup-chart value.
- Fighter order is never a feature. Features are differences (Fighter A minus Fighter B), the model is trained on both orderings of every fight, and test predictions are averaged over both orderings.

### 4.3 Step 1 – Clustering into styles
Standardise the 15 `pre_` style features and group fighter profiles with **K-Means** (default k = 4). Silhouette scores for k = 3 … 8 are printed to help pick k. We name each cluster by inspecting which features are high or low in its centre.

### 4.4 Step 2 – Style matchup chart
For every pair of styles, count how often fighters of one style beat fighters of the other, using training fights only. The resulting heatmap shows win rates with the number of fights in each cell, and marks cells that differ from 50% (binomial test, p < 0.05). Because many pairs are tested, we also report the stricter Bonferroni threshold. Thin cells are shrunk toward 50% when used as a model feature.

### 4.5 Step 3 – Does style help predict winners?
Logistic regression with three feature sets, compared on the test fights:

| Feature set | Contents |
|---|---|
| **Skill only** | Differences in win rate, finish rate, and experience |
| **Skill + style profile** | Above, plus differences in the 15 style features and each fighter's style cluster |
| **Skill + style + matchup chart** | Above, plus the historical win rate of fighter A's style against fighter B's style |

Reference point: a simple baseline that picks the fighter with the **better win record**. We report accuracy with 95% bootstrap confidence intervals, log-loss, and a **paired bootstrap** of the accuracy gain over the skill-only model, which tells us whether any improvement is larger than random noise.

---

## 5. Limitations

- **Statistics, not technique.** Styles are inferred from numbers; the data cannot tell a boxer from a Muay Thai fighter.
- **Small samples.** Many fighters have few UFC fights, so their style estimates are noisy, and some style pairs have few fights.
- **Opponent quality is ignored.** Strong numbers against weak opponents look the same as against elite ones.
- **Missing fields.** The datasets have no age, height, reach, or betting odds.
- **Snapshot data.** Nothing after December 2024; early UFC events (1990s) had different rules and less data.
- **Derived data.** The Hugging Face tables are copies of UFCStats and contain errors (like the mismatched fight IDs). We checked and fixed the one we found, but others may exist.

---

## 6. How to run the project

### Setup
Python 3.10+ (3.12 recommended). From the repository root:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
scipy
datasets
pyarrow
```

### Run (in order)

```bash
python load_data.py        # 1. downloads the two Hugging Face datasets -> data/raw/
python build_dataset.py    # 2. links them, builds pre-fight style features -> data/processed/
```

`build_dataset.py` options: `--window` (how many past fights to average, default 5) and `--min-prior` (minimum past fights required, default 2).

The first command needs an internet connection and takes about a minute. No scraping and no API keys are needed.

### Repository layout

```
.
├── README.md
├── requirements.txt
├── load_data.py            # download raw data
├── build_dataset.py        # link tables + build features
└── data/
    ├── raw/                # fights.csv, rounds.csv
    └── processed/          # fighter_fights.csv, fights_wide.csv
```

---

## 7. References

1. K. McQuaide, *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes*, Stanford (course project paper that inspired this work).
2. Hitkul et al., *A Comparative Study of Machine Learning Algorithms for Prior Prediction of UFC Fights*, Harmony Search and Nature Inspired Optimization Algorithms, 2018.
3. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.
4. UFCStats.com: original source of all fight statistics.
5. Hugging Face datasets `JesterLabs/UFC_FIGHT_DATA` and `JesterLabs/UFC_FIGHT_ROUNDS_DATA`.