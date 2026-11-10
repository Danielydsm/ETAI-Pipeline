"""Feature engineering: derived features added before the split.

Every function here computes a fixed formula -- nothing is learned from the data -- so,
like the cleaning rules in src/data.py, it is safe to run on the whole dataset and on
label-free inference data. A feature that needs a statistic (e.g. "above the median")
would instead have to be a step inside the model's Pipeline, so the statistic comes from
the training rows only.

See Practical/W6/notebooks/05_ensembles.ipynb, Section 6 ("Engineering a Feature for the
Trees") and Challenge 1.
"""
import pandas as pd


def add_engineered_features(df: pd.DataFrame, formulas: dict) -> pd.DataFrame:
    """Adds fixed-formula features (week 6 challenge), read from the config.

    Each entry of `formulas` is `{new_column: expression}`, evaluated with `df.eval()`.
    Fixed formulas only -- nothing is learned from the data -- so this is safe before the
    split and on new data. The new columns must also be listed in
    `preprocessing.numeric_features` (or `categorical_features`) so the ColumnTransformer
    picks them up.
    """
    out = df.copy()
    for name, expression in (formulas or {}).items():
        out[name] = out.eval(expression)
    return out
