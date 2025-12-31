from typing import Literal

import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


# shut up pycharm, I know variables shouldn't start with capitals.
# it's a matrix, and we do it like this.
# noinspection PyPep8Naming
def impute_column(
    df: DataFrame,
    col: str,
    strategy: Literal["mean", "median", "most_frequent", "sentinel", "model"] = "median",
    **kwargs,
):
    if col not in df.columns:
        print(f"Column {col} not found")
        return

    print(f"Imputing '{col}' using {strategy}")

    missing_before = df[col].isna().sum()
    print(f"Missing values before imputation: {missing_before}")

    # ----- Simple Strategies -----

    if strategy in {"mean", "median", "most_frequent"}:
        imputer = SimpleImputer(strategy=strategy)
        df[[col]] = imputer.fit_transform(df[[col]])

    # ----- Sentinel -----

    elif strategy == "sentinel":
        if np.issubdtype(df[col].dtype, np.integer):
            sentinel = kwargs.get("fill_value", -999)
        elif np.issubdtype(df[col].dtype, np.floating):
            sentinel = kwargs.get("fill_value", -999.0)
        else:
            sentinel = kwargs.get("fill_value", "MISSING")

        df[col] = df[col].fillna(sentinel)

    # ----- Model-Based -----

    elif strategy == "model":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != col]

        if not feature_cols:
            print("No other numeric columns to impute")
            return

        mask_missing = df[col].isna()
        mask_exist = ~mask_missing

        if mask_missing.sum() == 0:
            print(f"{col} has no missing values")
            return

        X_train = df[feature_cols][mask_exist]
        y_train = df[col][mask_exist]

        X_missing = df[feature_cols][mask_missing]

        rf = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 50),
            random_state=42,
        )

        rf.fit(X_train, y_train)
        df[col][mask_missing] = rf.predict(X_missing)

    else:
        print(f"Unknown strategy: {strategy}")

    missing_after = df[col].isna().sum()
    print(f"Missing values after imputation: {missing_after}")
