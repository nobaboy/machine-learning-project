from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.visualization import visualize_outlier_removal, visualize_numerical_correlation

__all__ = (
    "impute_column",
    "remove_outliers",
    "get_top_correlations",
    "multicollinearity",
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
        print(f"\nFailed to impute {col}, column is non-existent")
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


def get_top_correlations(
    df: DataFrame,
    feature_cols: list[str],
    top_n: int = 10,
    sample_size: int = 5000,
):
    if not feature_cols:
        return []

    if sample_size and 0 < sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 2:
        return []

    corr_matrix = df[numeric_cols].corr().abs()

    pairs = []
    cols = numeric_cols

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = corr_matrix.iloc[i, j]

            if corr > 0.1:
                pairs.append((cols[i], cols[j], corr))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def multicollinearity(
    df: DataFrame,
    feature_cols: list[str],
    corr_threshold: float = 0.85,
    sample_size: int = 10000,
    plot: bool = True,
):
    if not feature_cols:
        return [], [], []

    if sample_size and 0 < sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    numeric_cols = [col for col in feature_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if len(numeric_cols) < 2:
        print("Failed to check multicollinearity, not enough numeric features")
        return feature_cols, [], []

    variances = df[numeric_cols].var()
    numeric_cols = [col for col in numeric_cols if variances.get(col, 0.0) > 0.0]

    if len(numeric_cols) < 2:
        print("Failed to check multicollinearity, ...")
        return feature_cols, [], []

    print(f"\nChecking multicollinearity for {len(feature_cols)} numeric features")

    corr_matrix = df[numeric_cols].corr().abs()

    if plot:
        visualize_numerical_correlation(corr_matrix, title="Feature Correlation Heatmap (Before Removal)")

    # Find highly correlated pairs
    high_corr_pairs = []
    features_to_remove = set()

    cols = numeric_cols
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = corr_matrix.iloc[i, j]

            if np.isnan(corr) or corr <= corr_threshold:
                continue

            col1 = cols[i]
            col2 = cols[j]
            high_corr_pairs.append((col1, col2, corr))

            # Decide which to remove keep the one with higher variance
            var1 = df[col1].var()
            var2 = df[col2].var()

            if var1 >= var2:
                features_to_remove.add(col2)
                print(f"Removing {col2} (kept {col1}, corr={corr:.3f})")
            else:
                features_to_remove.add(col1)
                print(f"Removing {col1} (kept {col2}, corr={corr:.3f})")

    # Results
    features_to_keep = [col for col in feature_cols if col not in features_to_remove]

    if plot and 1 < len(features_to_keep) <= 20:
        # we only want to visualize the correlation of numerical features
        keep_numeric = [col for col in numeric_cols if col not in features_to_remove]
        visualize_numerical_correlation(
            df[keep_numeric].corr().abs(),
            title="Feature Correlation Heatmap (After Removal)",
            figsize=(10, 8),
        )

    if features_to_remove:
        print(f"\nRemoved features: {list(features_to_remove)}")

    return features_to_keep, list(features_to_remove), high_corr_pairs



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
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        print(f"Unexpected method: {method}")
        return df, None

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df, scaler
