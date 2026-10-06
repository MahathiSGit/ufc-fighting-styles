Do Fighting Styles Matter? A Style-Aware UFC Fight Predictor
Course: UE24CS352A – Machine Learning Mini-Project
Team: Mahathi S and Teammate
Topic: Predicting UFC fight outcomes and testing whether fighting style adds value beyond fighter skill
1. The idea
Most UFC fight predictors focus on skill-related information such as a fighter's record, Elo rating, striking accuracy, takedown ability, age, reach, and experience.
But MMA also has a matchup component. Different styles can interact differently: a wrestler/controller may perform differently against a striker than against another grappler.
This project asks whether that style information actually improves prediction.
We approach the question in three stages:
1. Build fighter profiles. Use historical UFC statistics to describe each fighter before a fight.
2. Identify fighting styles. Cluster fighters with similar pre-fight statistical profiles into interpretable style groups.
3. Test the value of style. Compare a skill-only model with models that additionally use style information.
The important part is that we do not assume style must improve the model. A small improvement, no improvement, or even a negative result is useful if the experiment is designed correctly.
2. Goal
Does knowing a fighter's style improve our ability to predict who wins a UFC fight, compared with skill-based information alone?

The project is designed as a chronological prediction problem: for every fight, the model should only use information that would have been available before that fight happened.
3. The data
3.1 Source
The project uses a public UFC dataset derived from UFCStats.com.
The raw project data contains:
- Historical UFC fights
- Fighter identities and metadata
- Fighter career/profile information
- Fight outcomes
- Fight statistics
The modelling pipeline uses the historical data to reconstruct what was known about each fighter immediately before each fight.
3.2 Prediction target
For each decisive fight:
- 1 = Fighter A wins
- 0 = Fighter B wins
Draws and no-contests are excluded from the binary classification task.
3.3 Pre-fight information
Examples of information used to build fighter profiles include:
Feature	Meaning
win_rate	Historical proportion of fights won
n_fights	Number of previous UFC fights
elo	Chronological Elo skill rating
streak	Current win/loss streak
layoff_days	Days since previous fight
age	Fighter age at the fight date
height	Fighter height
reach	Fighter reach
sig_acc	Significant-strike accuracy
sig_def	Significant-strike defense
td_acc	Takedown accuracy
td_def	Takedown defense
sig_diff_pm	Significant-strike differential per minute
kd15	Knockdowns per 15 minutes
kda15	Knockdowns absorbed per 15 minutes
ko_l_rate	Historical KO/TKO loss rate
sub_l_rate	Historical submission loss rate


4. Method
4.1 Preventing data leakage
The biggest methodological requirement is:
A fight's outcome must never be used to construct the features used to predict that same fight.

All historical fighter features are therefore calculated using only fights that occurred strictly earlier than the fight being predicted.
The project also avoids using raw career snapshot columns when those snapshots could contain future information.
4.2 Chronological train / validation / test split
The main experiment uses time-based splitting:
Split	Period	Fights
Training	2014–2022	4,336
Validation	2023	533
Test	2024–Aug 2026	1,429


This is closer to the real task of predicting future fights than randomly shuffling historical fights between train and test.
4.3 Fighter symmetry
The identity of Fighter A versus Fighter B should not affect the prediction.
Therefore:
- Features are represented as A − B differences
- The model is trained with both A/B and B/A versions
- Final predictions are averaged across both orderings
This prevents the model from learning an artificial advantage from fighter ordering.
5. Building fighting styles
5.1 Style features
A fighter's style is described using historical performance statistics available before the fight.
The clustering features include:
Feature	What it captures
slpm	Significant strikes landed per minute
head_share	Share of landed significant strikes aimed at the head
leg_share	Share aimed at the legs
clinch_share	Share landed in the clinch
ground_share	Share landed on the ground
td15	Takedowns per 15 minutes
sub15	Submission attempts per 15 minutes
ctrl_share	Fraction of fight time spent in control


5.2 Clustering
The style pipeline:
1. Build pre-fight fighter statistics.
2. Require sufficient historical data before assigning a known style.
3. Standardise the style variables.
4. Apply KMeans clustering with k = 5.
5. Assign each fighter to the nearest cluster at each fight.
6. Name the clusters from their statistical characteristics.
The five resulting styles are:
1. Volume striker
2. Leg-kick striker
3. Clinch grinder
4. Wrestler / controller
5. Submission hunter
5.3 Cluster quality
The clusters are useful for interpretation but are not perfectly separated.
- Silhouette score: approximately 0.17
- Bootstrap stability: approximately 0.70 adjusted Rand index
This means the project treats style as an approximate statistical description rather than a claim that every fighter belongs to one rigid category.
6. Style matchup analysis
After assigning styles, we examine how styles perform against one another.
Instead of only comparing raw win rates, the project measures a style's result relative to the Elo expectation:
Style edge = Actual result − Elo-expected result
This matters because raw style win rates can be misleading if one style happens to contain more elite fighters.
Main finding
The strongest repeated pattern was:
Wrestler / controllers outperformed their Elo expectation against every other style in both the training period and the 2024–2026 test period.

Across all modelling fights, wrestler/controllers showed approximately a:
+5.6 percentage-point edge versus Elo expectation.
The matchup matrix is treated as descriptive and exploratory, not as proof that one style causes another to lose.
7. Does style improve prediction?
This is the central experiment.
7.1 Models
Two model families are compared:
- Logistic Regression
- Histogram-based Gradient Boosting
7.2 Feature sets
Three versions are evaluated:
Skill only – 17 features
Uses historical fighter skill, experience, physical and performance variables.
Skill + style clusters – 24 features
Adds the categorical style information and style-matchup features.
Skill + style descriptors – 31 features
Adds continuous style measurements instead of relying only on hard cluster labels.
7.3 Test results
The final test set contains 1,429 unseen fights.
Model	Test accuracy	95% CI
Skill-only Logistic Regression	62.2%	59.6–64.7%
Skill + Style Clusters Logistic Regression	62.4%	59.9–64.9%
Skill + Style Descriptors Logistic Regression	63.3%	60.7–65.8%
Skill-only Gradient Boosting	60.0%	57.5–62.6%
Skill + Style Clusters Gradient Boosting	59.8%	57.3–62.4%
Skill + Style Descriptors Gradient Boosting	61.2%	58.6–63.7%


The hard style-cluster model improves logistic-regression accuracy by only:
+0.2 percentage points
with a 95% confidence interval of:
−1.0 to +1.3 percentage points
7.4 Interpretation
The result is important:
There is no reliable evidence that adding hard fighting-style clusters improves prediction beyond skill features.

Continuous style descriptors show a slightly larger point estimate, but the confidence interval still includes zero.
In other words:
- Style can reveal interesting matchup patterns.
- Style can describe how fighters behave statistically.
- But that does not automatically mean style makes the prediction model more accurate.
That distinction is a major part of this project.
8. Model evaluation and uncertainty
Accuracy is reported with 95% bootstrap confidence intervals.
The model comparison also examines log-loss and the change in accuracy relative to the skill-only baseline.
The test set is kept separate from model and feature selection as much as possible within the experiment.
9. Applications
9.1 Streamlit web app
Run:
streamlit run app/style_app.py
The application allows a user to select two fighters and a fight date.
It displays:
- Fighter styles
- Skill-only prediction
- Skill + style prediction
- Difference between predictions
- Historical style-matchup edge
- Matchup sample size
- Style profiles
- Important prediction factors
- Side-by-side fighter statistics
9.2 Command-line predictor
Example:
python app/style_predictor.py "Islam Makhachev" "Charles Oliveira"
The predictor returns the available model probabilities and relevant style information.
Note: This is an academic machine-learning project, not betting advice.

10. Project structure
ufc-fighting-styles/
│
├── app/
│   ├── style_app.py
│   └── style_predictor.py
│
├── data/
│   ├── raw/
│   │   └── master.csv
│   │
│   └── processed/
│       ├── fight_features_style.csv
│       ├── fighter_styles.csv
│       ├── fighter_latest_state.csv
│       └── style_profiles.csv
│
├── modeling/
│   ├── style_clusters.py
│   ├── compare_models.py
│   └── train_style_models.py
│
├── models/
│   └── style_models.joblib
│
├── preprocessing/
│   └── build_dataset.py
│
├── reports/
│   ├── model_comparison.csv
│   └── figures/
│       ├── style_matrix.png
│       ├── style_profiles.png
│       └── model_comparison.png
│
├── tests/
│   ├── test_style_predictor.py
│   ├── test_no_leakage.py
│   └── test_predictor.py
│
├── requirements.txt
└── README.md
11. How to run the project
Setup
Windows:
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Run the pipeline
python preprocessing/build_dataset.py
python modeling/style_clusters.py
python modeling/compare_models.py
python modeling/train_style_models.py
python tests/test_style_predictor.py
streamlit run app/style_app.py
For the original/base predictor:
python modeling/train_models.py
12. Testing and validation
The project contains dedicated checks for correctness and leakage.
Style validation
tests/test_style_predictor.py
Reconstructs fighter states at multiple dates and checks that the resulting style labels agree with the experiment output.
63 real fighter-fight checks passed with 0 mismatches.
Leakage validation
tests/test_no_leakage.py
Checks that:
- historical features are rebuilt correctly,
- a fight's own result cannot affect its own features,
- the dataset passes basic balance and sanity checks.
Predictor consistency
tests/test_predictor.py
Checks that application feature generation reproduces the features used during training.
13. Limitations
Statistical limitations
- Style clusters have weak separation (silhouette ≈ 0.17).
- Style is unknown for a substantial fraction of historical fights.
- Some style matchup cells have relatively small sample sizes.
- The 1,429-fight test set has limited power to detect very small accuracy differences.
- No multiple-comparison correction was applied to individual matchup cells.
Data limitations
- The dataset is limited to UFC information available through August 2026.
- Injuries and training-camp information are not included.
- Betting odds are not included as a benchmark.
- Pre-UFC records are not included.
- Opponent quality is not completely controlled for outside the Elo-based analysis.
Interpretation limitations
- A style edge is an historical association, not a causal effect.
- A fighter can change style over a career.
- Hard cluster labels simplify a continuous range of fighting behaviours.
14. Future work
Possible extensions include:
- Soft style membership instead of hard KMeans labels
- Larger rolling-origin backtests
- Style × style interaction features
- Round-level statistics
- Opponent-adjusted performance metrics
- Weight-class-specific styles
- Betting odds as an external benchmark
- Per-style models
15. References
1. McQuaide, K., Applying Machine Learning Algorithms to Predict UFC Fight Outcomes, Stanford.
2. UFCStats.com – source of UFC fight statistics.
3. Scikit-learn documentation and algorithms used in the project.
16. Conclusion
This project investigates an intuitive idea in MMA: does it matter not only how good a fighter is, but what kind of fighter they are facing?
The results show two different things.
First, fighting styles reveal meaningful descriptive patterns. In particular, wrestler/controllers consistently outperform their Elo expectation against several opposing styles.
Second, those patterns do not translate into a reliable improvement in overall prediction accuracy when hard style clusters are added to the model.
The final conclusion is therefore:
Fighting style is interesting and informative as an analytical lens, but this experiment does not provide reliable evidence that hard style categories improve UFC fight prediction beyond skill-based features.
