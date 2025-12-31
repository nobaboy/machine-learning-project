from pandas import DataFrame

from old_utils import (
    remove_outliers,
    featuresEng,
    get_feature_names,
    feature_scaling,
    get_top_correlations,
    multicollinearity,
)
from utils.loader import load_data, calculate_memory_usage, format_memory_size, optimize_memory_usage
from utils.visualization import visualize_memory_usage, analyze_and_visualize_missing, plot_top_correlations
from utils.preprocessing import impute_column


def main():
    datasets: dict[str, DataFrame] = {
        "aisles": load_data("data/aisles.csv"),
        "departments": load_data("data/departments.csv"),
        "products": load_data("data/products.csv"),
        "orders": load_data("data/orders.csv"),
        "prior": load_data("data/order_products__prior.csv"),
        "train": load_data("data/order_products__train.csv"),
    }

    # ----- Optimization -----

    mem_usage_before = calculate_memory_usage(*datasets.values())

    for name, dataset in datasets.items():
        optimize_memory_usage(name, dataset)

    mem_usage_after = calculate_memory_usage(*datasets.values())
    pct = 100 * (mem_usage_before - mem_usage_after) / mem_usage_before
    print(f"\nReduced total memory usage from {format_memory_size(mem_usage_before)} to {format_memory_size(mem_usage_after)} ({pct:.1f}% reduction)")

    visualize_memory_usage(mem_usage_before, mem_usage_after)

    aisles, departments, products = datasets["aisles"], datasets["departments"], datasets["products"]
    orders, prior, train = datasets["orders"], datasets["prior"], datasets["train"]

    print("\nMerging datasets...")

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

    missing_cols = analyze_and_visualize_missing(data_full)

    # TODO this imputes one column here, it being days_since_prior_order, imo it should be set to 0 rather than
    #      median as it could be the *first* order for a user, which is why it's missing
    for col in missing_cols:
        impute_column(data_full, col)

    # ----- Outlier Handling -----

    # TODO why these 2 columns only
    columns_to_check = ['days_since_prior_order', 'add_to_cart_order']
    # TODO the professor suggested winsorizing, check that out instead of removing outliers
    data_full = remove_outliers(data_full, columns_to_check)

    # TODO temporary return
    return

    # ----- Create train labels -----

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
    train_with_labels["reordered"] = train["reordered"].fillna(0).astype("int8")

    # ----- Feature Engineering -----

    print(" FEATURE ENGINEERING")

    # Extract just prior data
    prior_only = prior.merge(
        orders[['order_id', 'user_id', 'order_number']],
        on='order_id'
    )

    # Create complete features
    engineered_train = featuresEng(
        prior_df=prior_only,
        orders_df=orders,
        train_df=train_with_labels,
    )

    # Get ALL feature names
    feature_cols = get_feature_names(engineered_train)

    print(f"\n Original features ({len(feature_cols)} total):")
    for i, col in enumerate(feature_cols[:10], 1):  # Show first 10
        print(f"{i:2}. {col}")
    if len(feature_cols) > 10:
        print(f"... and {len(feature_cols) - 10} more")

    # ========== FAST MULTICOLLINEARITY CHECK ==========
    print("\n" + "=" * 50)
    print("FAST MULTICOLLINEARITY CHECK")
    print("=" * 50)

    # 1. Show top correlations (optional, for insight)
    top_corrs = get_top_correlations(engineered_train, feature_cols, top_n=5)

    # 2. Remove highly correlated features WITH VISUALIZATION
    clean_features, removed_features, high_corr_pairs = multicollinearity(
        df=engineered_train,
        feature_cols=feature_cols,
        corr_threshold=0.85,  # Remove if correlation > 0.85
        sample_size=5000,  # Use 5k samples for speed
        plot=True  # Show correlation heatmaps
    )

    # 3. Show bar chart of removed correlations
    if high_corr_pairs:
        print("\n Visualization of removed correlations:")
        plot_top_correlations(high_corr_pairs, top_n=min(10, len(high_corr_pairs)))

    # Update feature list
    feature_cols = clean_features

    # ----- Scale the features -----
    print("\n" + "=" * 50)
    print("SCALING FEATURES")
    print("=" * 50)

    # Scale ONLY the clean features
    engineered_train_scaled, scaler = feature_scaling(
        df=engineered_train,
        columns_to_exclude=['user_id', 'product_id', 'order_id', 'reordered'] + removed_features,
    )

    # Ready for modeling
    X = engineered_train_scaled[feature_cols]
    y = engineered_train_scaled['reordered']

    # Show final features
    print(f"\n Final features for modeling ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2}. {col}")


if __name__ == "__main__":
    main()
