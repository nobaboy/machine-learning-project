from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.visualization import visualize_outlier_removal

__all__ = (
    "impute_column",
    "remove_outliers",
    "scale_features",
)


# shut up pycharm, I know variables shouldn't start with capitals.
# it's a matrix, and we do it like this.
# noinspection PyPep8Naming
def impute_column(
    df: DataFrame,
    col: str,
    strategy: Literal["mean", "median", "most_frequent", "sentinel", "model"] = "median",
    **kwargs,
):
    df = df.copy()

    if col not in df.columns:
        print(f"\nColumn {col} not found") # TODO better message
        return df

    print(f"\nImputing '{col}' using {strategy}")

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
            print("Data has no numeric columns to impute")
            return df

        mask_missing = df[col].isna()
        mask_exist = ~mask_missing

        if mask_missing.sum() == 0:
            print(f"{col} has no missing values")
            return df

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
        print(f"Unexpected strategy: {strategy}")
        return df

    missing_after = df[col].isna().sum()
    print(f"Missing values after imputation: {missing_after}")

    return df


def remove_outliers(df: DataFrame, cols: list[str], plot: bool = True):
    df = df.copy()

    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        print(f"\nProcessing: {col}")

        # Statistics
        before_count = len(df)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            print(f"Skipping {col} since it has no variation")
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Define what to keep and what to remove
        mask_keep = (df[col] >= lower) & (df[col] <= upper)
        outliers_count = (~mask_keep).sum()  # Count the inverse True outliers

        if plot:
            visualize_outlier_removal(df, col, mask_keep, outliers_count, before_count)

        df = df[mask_keep].copy()

        pct = 100 * outliers_count / before_count
        print(f"Removed {outliers_count} outliers ({pct:.1f}%)")

    return df


def scale_features(
    df: DataFrame,
    excluded_columns: list[str],
    method: Literal["standard", "minmax"] = "minmax",
):
    # just use number here to also scale the new engineered features since their dtypes
    # aren't the optimized variant
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [col for col in numeric_cols if col not in excluded_columns]

    if not cols_to_scale:
        return df, None

    print(f"\nScaling {len(cols_to_scale)} numeric columns: ")
    print(f"Columns to scale: {cols_to_scale}")

    if method == "standard":
        scaler = StandardScaler() # I'm using Logistic Regression and SVM work better with StandardScaler
    elif method == "minmax":
        scaler = MinMaxScaler() # K-Nearest Neighbors KNN - distance based
    else:
        print(f"Unexpected method: {method}")
        return df, None

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df, scaler
