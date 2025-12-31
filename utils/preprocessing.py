from typing import Literal

import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from utils.visualization import visualize_outlier_removal


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


def remove_outliers(df: DataFrame, cols: list[str], plot: bool = True):
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
