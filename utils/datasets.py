from pandas import DataFrame
from sklearn.model_selection import GroupShuffleSplit

from utils.feature import build_user_features
from utils.preprocessing import scale_features, remove_multicollinearity
from utils.visualization import visualize_top_correlations

__all__ = (
    "prep_task_a_dataset",
    "prep_task_b_dataset",
)


def group_split(df: DataFrame, test_size: float =0.25, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["user_id"]))

    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    return train, test


# shut up pycharm again
# noinspection PyPep8Naming
def prep_task_a_dataset(
    engineered_train: DataFrame,
    feature_cols: list[str],
    corr_threshold: float = 0.85,
    sample_size: int = 5000,
    test_size: float = 0.25,
    random_state: int = 42,
):
    print("\n" + "=" * 30)
    print("SPLITTING")
    print("=" * 30)

    _train, _test = group_split(engineered_train, test_size=test_size, random_state=random_state)

    print("\n" + "=" * 30)
    print("MULTICOLLINEARITY CHECK")
    print("=" * 30)

    print(f"\nNumber of features before removing multicollinearity: {len(feature_cols)}")

    clean_features, removed_features, high_corr_pairs = remove_multicollinearity(
        df=_train,
        feature_cols=feature_cols,
        corr_threshold=corr_threshold,
        sample_size=sample_size,
        plot=True,
    )

    if high_corr_pairs:
        print("\nVisualization of removed correlations:")
        visualize_top_correlations(high_corr_pairs, top_n=min(10, len(high_corr_pairs)))

    feature_cols = clean_features
    _test = _test.drop(columns=removed_features, errors="ignore")

    print(f"\nNumber of features after removing multicollinearity: {len(feature_cols)}")

    print("\n" + "=" * 30)
    print("SCALING FEATURES")
    print("=" * 30)

    excluded = ["user_id", "product_id", "order_id", "reordered"] + removed_features

    scaled_train, scalar = scale_features(df=_train, excluded_columns=excluded)
    scaled_test, _ = scale_features(df=_test, excluded_columns=excluded, scalar=scalar)

    X_train = scaled_train[feature_cols]
    y_train = scaled_train["reordered"]

    X_test = scaled_test[feature_cols]
    y_test = scaled_test["reordered"]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test


# shut up pycharm again v2
# noinspection PyPep8Naming
def prep_task_b_dataset(
    prior: DataFrame,
    orders: DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
):
    train_orders = orders.loc[orders["eval_set"] == "train", ["user_id", "days_since_prior_order"]].copy()
    train_orders = train_orders.dropna(subset=["days_since_prior_order"])

    prior_user = prior.merge(
        orders[["order_id", "user_id", "order_number", "days_since_prior_order", "order_dow", "order_hour_of_day"]],
        on="order_id",
        how="left",
    )
    X_user = build_user_features(prior_user)

    reg_df = X_user.merge(train_orders, on="user_id")

    _train, _test = group_split(reg_df, test_size=test_size, random_state=random_state)

    excluded = ["user_id", "order_id", "days_since_prior_order"]
    scaled_train, scalar = scale_features(df=_train, excluded_columns=excluded)
    scaled_test, _ = scale_features(df=_test, excluded_columns=excluded, scalar=scalar)

    X_train = scaled_train.drop(columns=["days_since_prior_order"])
    y_train = scaled_train["days_since_prior_order"]

    X_test = scaled_test.drop(columns=["days_since_prior_order"])
    y_test = scaled_test["days_since_prior_order"]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test
