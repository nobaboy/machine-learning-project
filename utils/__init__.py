from pandas import DataFrame

__all__ = ("get_feature_names",)

EXCLUDED_COLUMNS: set[str] = {"user_id", "product_id", "order_id", "reordered"}


def get_feature_names(
    df: DataFrame,
) -> list[str]:
    feature_cols = [col for col in df.columns if col not in EXCLUDED_COLUMNS]

    print(f"\nFeature columns ({len(feature_cols)} total):")
    for col in feature_cols:
        print(f" - {col}")

    return feature_cols
