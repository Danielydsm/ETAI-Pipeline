# Baseline Predictive Pipeline -- ETAI

*Daniely da Silva Machado - 20260617*

This is the **starting point** for my semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # metrics from probabilities + fairness check
│   ├── tuning.py           # Optuna hyperparameter search + nested CV
│   └── results.py          # saves each run's report + the scoreboard
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | Data diagnosis (EDA) | `clean_dataset()` driven by the diagnosis written as rules in `config.yaml` (`validity_rules`, `canonical_categories`, `placeholder_tokens`, `redundant_columns`): category cleanup, impossible values / placeholders → `NaN`, redundant-column removal |
| 4 | Leak-safe preprocessing + honest evaluation | Row-preserving cleaning + training-only `drop_duplicate_rows()`; `split_dev_test()` (locked test set) replacing `split_train_test`; `build_preprocessor()` (mechanism-matched imputation, target encoding, robust scaling — all fitted inside the `Pipeline`); stratified 5-fold CV (`cross_validate_pipeline`, out-of-fold classification report + fairness check); `dummy` and `random_forest` models |
| 7 | Boosting + evaluation by probabilities | `gradient_boosting`, `hist_gradient_boosting` and `naive_bayes` models; eight metrics scored from out-of-fold **probabilities** (ROC-AUC is the selection metric, chosen before comparing models); a **scoreboard** (`results/runs.csv`) keeps every run with the no-model floor and the untuned score; the corrected paired t-test separates real differences from noise |

## Model evaluation

The pipeline no longer trusts a single train/test split. It carves a **locked test set** aside once (20%, stratified, seed 42 — never used to fit or choose), and judges every candidate with **stratified 5-fold cross-validation** on the development set, fitting the *whole* pipeline (preprocessing + model) inside each fold so no fold leaks into its own preprocessing.

| Model | Holdout accuracy (1 split) | Holdout range, 30 seeds | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|---|
| Dummy (majority) | 0.550 | 0.550 – 0.550 | 0.549 ± 0.000 | 0.000 |
| Logistic regression | 0.672 | 0.658 – 0.697 | 0.672 ± 0.013 | +0.003 |
| Decision tree | 0.602 | 0.558 – 0.624 | 0.611 ± 0.015 | +0.084 |
| Random forest | 0.655 | 0.623 – 0.674 | 0.652 ± 0.017 | +0.080 |

**Which number to trust: CV.** A single holdout is one draw from a distribution — for logistic regression the same model moves from 0.658 to 0.697 (~4 points) just by changing the seed, so a lone 0.672 is partly luck. CV uses every development row for validation and reports the mean and its spread (std), so it is both more stable and more honest. The week 2/3 "best model" conclusion still holds under CV: logistic regression remains the best model (highest CV accuracy) and the least overfit (gap ≈ 0 vs ~0.08 for the tree and forest).

Changing the scaler (`none` / `standard` / `minmax` / `robust`) leaves CV at ~0.672 (±0.013) — the difference is far smaller than the CV noise, so `robust` stays (a safe default for the skewed counts). KNN imputation was also tried and was slightly *worse* (0.669 vs 0.672) — the `_was_missing` flags already carry the missingness signal, so the median fill stays.

## Scoreboard (Week 7)

From week 7 the pipeline reports **eight metrics** computed from out-of-fold **probabilities**
(not a single 0/1 prediction), and optimises one **selection metric**: **F1**. F1 (the harmonic
mean of precision and recall) is the choice because recidivism has a clear positive class to
catch, both false positives and false negatives matter, and accuracy is a poor guide on this
imbalanced target — so a single balanced number that rewards detecting reoffenders (recall)
without flooding the flagged set (precision) beats threshold-free ranking.

Every `python main.py` run appends one row to `results/runs.csv` with the model, its settings,
the selection metric's mean/std and per-fold scores, plus the no-model floor (a dummy) and the
untuned score. Nested-CV (tuned) results, best first:

| Model | Tuned | F1 (mean ± std) | Untuned | Floor |
|---|---|---|---|---|
| HistGradientBoosting | ✓ | 0.644 ± 0.018 | 0.630 | 0.000 |
| Random forest | ✓ | 0.639 ± 0.018 | 0.606 | 0.000 |
| Logistic regression | ✓ | 0.633 ± 0.012 | 0.634 | 0.000 |
| Decision tree | ✓ | 0.633 ± 0.019 | 0.567 | 0.000 |
| Gradient boosting | ✓ | 0.620 ± 0.021 | 0.607 | 0.000 |
| AdaBoost | ✓ | 0.619 ± 0.016 | 0.599 | 0.000 |
| Oblique random forest | ✓ | 0.616 ± 0.011 | 0.596 | 0.000 |
| Naive Bayes | ✓ | 0.404 ± 0.013 | 0.403 | 0.000 |
| Dummy (floor) | ✗ | 0.000 ± 0.000 | 0.000 | 0.000 |

The corrected paired t-test on the same folds says HistGradientBoosting is not significantly
better than random forest, logistic regression or the decision tree, but it is better than
gradient boosting (p ≈ 0.04), AdaBoost and the oblique forest (p ≈ 0.02), and far better than
Naive Bayes (p ≈ 0.0001).

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
