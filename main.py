"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean -> drop duplicate rows (training only)
    -> split features/target -> lock the test set
    -> preprocessing + model as ONE Pipeline (src/model.py)
    -> stratified k-fold cross-validation of the whole pipeline (week 4)
    -> hyperparameter tuning with Optuna, leak-free, on the development set only (week 5)
    -> out-of-fold classification report + fairness check
    -> refit the final (tuned) model on the whole development set -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test,
)
from src.model import build_pipeline
from src.evaluate import (
    cross_validate_pipeline, cv_report, oof_classification_report, fairness_report,
)
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # load + diagnose-and-clean: nothing here is learned from the data, so it's safe to run
    # on the whole dataset -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    # training data only: the same person must not count twice, or sit in both dev and test
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # the final test set is set aside HERE and never used again in this script
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # preprocessing + model = one estimator: anything that learns is re-fit in every fold
    pipeline = build_pipeline(config["preprocessing"], config["model"])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    # week 4: default-hyperparameter CV of the whole pipeline
    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs)
    report = cv_report(fold_scores, scoring)

    # week 5: hyperparameter tuning -- development set only, every candidate scored by CV of
    # the whole pipeline (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    if tuning_config.get("enabled", False):
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"No search space for model type '{model_type}'. "
                             f"Available: {sorted(search_spaces)}")
        search_space = search_spaces[model_type]
        n_trials = tuning_config.get("n_trials", 20)
        tuning_seed = tuning_config.get("random_state", config["test_set"]["random_state"])
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True,
                                   random_state=tuning_seed)

        # 1. the honest estimate of the procedure (its out-of-fold predictions feed the reports)
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs
        )
        report += "\n\n" + cv_report(nested_scores, scoring)

        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs
        )
        report += "\n" + tuning_report(study, nested_scores, scoring)

    # reports from the (tuned) out-of-fold predictions
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # the model we'd actually use: same pipeline, refit on EVERY development row
    final_model = pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    print(refit)
    report += "\n" + refit + "\n"

    locked = (f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
              f"Development set: {len(X_dev)} rows.")
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
