from pandas import DataFrame

from utils import get_feature_names
from utils.feature import create_features
from utils.loader import load_data, calculate_memory_usage, format_memory_size, optimize_memory_usage
from utils.visualization import visualize_memory_usage, analyze_and_visualize_missing, visualize_top_correlations
from utils.preprocessing import impute_column, remove_outliers, get_top_correlations, multicollinearity, scale_features


def main():
    print("=" * 30)
    print("LOADING DATASETS")
    print("=" * 30 + "\n")

    datasets: dict[str, DataFrame] = {
        "aisles": load_data("data/aisles.csv"),
        "departments": load_data("data/departments.csv"),
        "products": load_data("data/products.csv"),
        "orders": load_data("data/orders.csv"),
        "prior": load_data("data/order_products__prior.csv"),
        "train": load_data("data/order_products__train.csv"),
    }

    # ----- Optimization -----

    print("\n" + "=" * 30)
    print("MEMORY OPTIMIZATION")
    print("=" * 30 + "\n")

    mem_usage_before = calculate_memory_usage(*datasets.values())

    for name, dataset in datasets.items():
        optimize_memory_usage(name, dataset)

    mem_usage_after = calculate_memory_usage(*datasets.values())
    pct = 100 * (mem_usage_before - mem_usage_after) / mem_usage_before
    print(f"\nReduced total memory usage from {format_memory_size(mem_usage_before)} to {format_memory_size(mem_usage_after)} ({pct:.1f}% reduction)")

    visualize_memory_usage(mem_usage_before, mem_usage_after)

    aisles, departments, products = datasets["aisles"], datasets["departments"], datasets["products"]
    orders, prior, train = datasets["orders"], datasets["prior"], datasets["train"]

    print("\n" + "=" * 30)
    print("MERGING DATASETS FOR EDA")
    print("=" * 30)

    print("Merging all 5 datasets...")

    # we use this for most of the eda we have to do
    data_full = (
        prior
        .merge(orders, on="order_id")
        .merge(
            products
            .merge(aisles, on="aisle_id")
            .merge(departments, on="department_id"),
            on="product_id"
        )
    )  # final version

    # ----- Imputation -----

    print("\n" + "=" * 30)
    print("MISSING VALUES + IMPUTATION")
    print("=" * 30)

    missing_cols = analyze_and_visualize_missing(data_full)

    for col in missing_cols:
        if col == "days_since_prior_order":
            data_full = impute_column(data_full, col, strategy="sentinel", fill_value=0)
        else:
            data_full = impute_column(data_full, col) # median

    # ----- Outlier Handling -----

    print("\n" + "=" * 30)
    print("OUTLIER HANDLING")
    print("=" * 30)

    cols_to_check = ["days_since_prior_order", "add_to_cart_order"]
    # TODO the professor suggested winsorizing, check that out instead of removing outliers
    data_full = remove_outliers(data_full, cols_to_check)

    # ----- Create train labels -----

    print("\n" + "=" * 30)
    print("BUILDING TRAIN LABELS")
    print("=" * 30)

    base_features = data_full[["user_id", "product_id"]].drop_duplicates()

    train_labels = train[["order_id", "product_id", "reordered"]]
    train_user = train_labels.merge(orders[["order_id", "user_id"]], on="order_id")

    train_with_labels = base_features.merge(
        train_user[["user_id", "product_id", "reordered"]],
        on=["user_id", "product_id"],
        how="left", # keep all the pairs from prior
    )

    # keeping all the pairs from prior introduces NaN values, which in python is a floating point number
    # so fill those missing values with 0 and optimize the dtype again
    train_with_labels["reordered"] = train_with_labels["reordered"].fillna(0).astype("int8")

    # ----- Feature Engineering -----

    print("\n" + "=" * 30)
    print("FEATURE ENGINEERING")
    print("=" * 30)

    prior_orders = prior.merge(
        orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="left",
    )

    engineered_train = create_features(
        prior=prior_orders,
        # orders=orders,
        train_pairs=train_with_labels[["user_id", "product_id", "reordered"]],
    )

    # Get ALL feature names
    feature_cols = get_feature_names(engineered_train)

    print(f"\nOriginal features ({len(feature_cols)} total):")
    for i, col in enumerate(feature_cols[:10], 1):  # Show first 10
        print(f"{i:>2}. {col}")
    if len(feature_cols) > 10:
        print(f"... and {len(feature_cols) - 10} more")

    # ----- Correlation and Multicollinearity -----

    print("\n" + "=" * 30)
    print("FAST MULTICOLLINEARITY CHECK")
    print("=" * 30)

    # 1. Show top correlations (optional, for insight)
    top_corrs = get_top_correlations(engineered_train, feature_cols, top_n=5)

    if top_corrs:
        print("\nTop correlations (sample):")
        for f1, f2, corr in top_corrs:
            print(f" - {f1} vs {f2}: {corr:.3f}")
        print()

    # TODO this removes some of the interaction features we made, maybe make a feature priority list?
    # 2. Remove highly correlated features WITH VISUALIZATION
    clean_features, removed_features, high_corr_pairs = multicollinearity(
        df=engineered_train,
        feature_cols=feature_cols,
        corr_threshold=0.85, # remove if correlation > 0.85
        sample_size=5000, # use 5k samples for speed
        plot=True, # show correlation heatmaps
    )

    # 3. Show bar chart of removed correlations
    if high_corr_pairs:
        print("\nVisualization of removed correlations:")
        visualize_top_correlations(high_corr_pairs, top_n=min(10, len(high_corr_pairs)))

    # Update feature list
    feature_cols = clean_features

    # ----- Scaling -----

    print("\n" + "=" * 30)
    print("SCALING FEATURES")
    print("=" * 30)

    # Scale ONLY the clean features
    excluded = ["user_id", "product_id", "order_id", "reordered"] + removed_features
    engineered_train_scaled, scaler = scale_features(
        df=engineered_train,
        excluded_columns=excluded,
    )

    # shut up pycharm again
    # noinspection PyPep8Naming
    X = engineered_train_scaled[feature_cols]
    y = engineered_train_scaled["reordered"]

    # Show final features
    print(f"\nFinal features for modeling ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2}. {col}")

    # quick info about the training matrix, remove after modelling

    print("\n" + "=" * 30)
    print("TRAINING MATRIX RESULT")
    print("=" * 30 + "\n")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Target values:\n{y.value_counts(normalize=True).rename("pct").mul(100).round(2)}")


if __name__ == "__main__":
    main()
