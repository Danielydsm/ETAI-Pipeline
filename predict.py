"""Prediction: load a trained model and predict unseen data.

Run with:
    python predict.py                     # config.yaml (no prediction section -> no-op)
    python predict.py config_adult.yaml

Generic: everything is read from the config's `prediction` section (the trained model's
path, the new data, the output path), so it works for any config that has one. `main.py`
also calls it automatically after training when a config declares a `prediction` section.
"""
import argparse
import os

import joblib
import pandas as pd
import yaml

from src.data import clean_dataset, load_data
from src.features import add_engineered_features
from src.preprocessing import split_features_target


def predict(config_path: str) -> str:
    """Loads the model and predicts the unseen rows named by the config's `prediction` section.

    Cleaning is identical to training, but `drop_duplicate_rows()` is never called here:
    every unseen row needs a prediction (rows out == rows in). Returns the output path.
    """
    config = yaml.safe_load(open(config_path))
    prediction = config.get("prediction")
    if not prediction:
        print(f"'{config_path}' has no `prediction` section -- nothing to predict.")
        return None

    model = joblib.load(prediction["model_path"])
    id_col = config["diagnostics"].get("id_column", "id")

    new_raw = load_data(prediction["new_data_path"])
    new_clean = clean_dataset(new_raw, config["diagnostics"])
    new_clean = add_engineered_features(new_clean, config["preprocessing"].get("engineered_features"))
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X_new, _, _ = split_features_target(new_clean, config["data"], mnar_sources)

    proba = model.predict_proba(X_new)[:, 1]
    threshold = prediction.get("threshold", 0.5)
    prediction_col = (proba >= threshold).astype(int)

    out = pd.DataFrame({"id": new_clean[id_col], "probability": proba, "prediction": prediction_col})
    os.makedirs(os.path.dirname(prediction["output_path"]) or ".", exist_ok=True)
    out.to_csv(prediction["output_path"], index=False)
    print(f"Predicted {len(out)} new people -> {prediction['output_path']} (rows out = rows in)")
    return prediction["output_path"]


def main():
    parser = argparse.ArgumentParser(description="Predict unseen data with a trained model.")
    parser.add_argument("config", nargs="?", default="config.yaml",
                        help="path to a config YAML (default: config.yaml)")
    args = parser.parse_args()
    predict(args.config)


if __name__ == "__main__":
    main()
