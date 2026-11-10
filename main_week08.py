"""
Entry point for week 9, Option A: run the pipeline on the Adult income dataset.

    python main_week08.py

Reuses the same orchestration as `main.py` (`run_pipeline`), but on `config_adult.yaml`,
then -- after the final model is refit on the whole development set -- predicts the unseen
people in `adult_new.csv` (same cleaning, no de-duplication, no target column) and saves
`id`, probability and 0/1 prediction to a CSV.

The COMPAS run (`python main.py`) is unchanged.
"""
import pandas as pd

from main import run_pipeline
from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target


def predict_new_people(config, pipeline):
    """Predicts the unseen rows (label-free) and writes id/probability/prediction to CSV.

    Cleaning is identical to training, but `drop_duplicate_rows()` is NEVER called here:
    every unseen row needs a prediction (rows out == rows in).
    """
    new_raw = load_data(config["prediction"]["new_data_path"])
    new_clean = clean_dataset(new_raw, config["diagnostics"])

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X_new, _, _ = split_features_target(new_clean, config["data"], mnar_sources)

    proba = pipeline.predict_proba(X_new)[:, 1]
    prediction = (proba >= 0.5).astype(int)

    out = pd.DataFrame({
        "id": new_clean["id"],
        "probability": proba,
        "prediction": prediction,
    })
    out.to_csv(config["prediction"]["output_path"], index=False)
    return config["prediction"]["output_path"], len(out)


def main():
    config, _, pipeline = run_pipeline("config_adult.yaml")

    path, n_rows = predict_new_people(config, pipeline)
    print(f"\nPredicted {n_rows} new people -> {path} (rows out = rows in)")


if __name__ == "__main__":
    main()
