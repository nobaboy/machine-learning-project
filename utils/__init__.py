import pandas as pd
from pandas import DataFrame

__all__ = (
    "get_feature_names",
    "get_top_correlations",
)

EXCLUDED_COLUMNS: set[str] = {"user_id", "product_id", "order_id", "reordered"}


def get_feature_names(df: DataFrame) -> list[str]:
    feature_cols = [col for col in df.columns if col not in EXCLUDED_COLUMNS]

    print(f"\nFeature columns ({len(feature_cols)} total):")
    for col in feature_cols:
        print(f" - {col}")

    return feature_cols


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
