from pandas import DataFrame
from sklearn.model_selection import train_test_split

from models import (
    linear_regressor,

    lasso_classifier,
    ridge_classifier,
    knn_classifier,
    svm_linear_classifier,
    svm_kernel_classifier,
    decision_tree_classifier,
    random_forest_classifier,
    xgboost_classifier,
    lightgbm_classifier,

    lasso_regressor,
    ridge_regressor,
    elastic_net_regressor,
    svm_linear_regressor,
    knn_regressor,
    decision_tree_regressor,
    random_forest_regressor,
    xgboost_regressor,
    lightgbm_classifier,
    lightgbm_regressor,
)

from utils import get_feature_names, get_top_correlations
from utils.evaluation import evaluate_classifier, evaluate_regressor, regression_accuracy
from utils.feature import features
from utils.loader import load_data, calculate_memory_usage, format_memory_size, optimize_memory_usage
from utils.preprocessing import impute_column, winsorize_outliers, remove_multicollinearity, scale_features
from utils.visualization import visualize_memory_usage, analyze_and_visualize_missing, visualize_top_correlations


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
    print("=" * 30 + "\n")

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
    )

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
    data_full = winsorize_outliers(data_full, cols_to_check)

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

    engineered_train = features(
        prior=prior_orders,
        # orders=orders,
        train_pairs=train_with_labels[["user_id", "product_id", "reordered"]],
    )

    # Get ALL feature names
    feature_cols = get_feature_names(engineered_train)

    print(f"\nOriginal features ({len(feature_cols)} total):")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"{i:>2}. {col}")
    if len(feature_cols) > 10:
        print(f"... and {len(feature_cols) - 10} more")

    # ----- Correlation -----

    print("\n" + "=" * 30)
    print("CORRELATION CHECK")
    print("=" * 30)

    # 1. Show top correlations
    top_corrs = get_top_correlations(engineered_train, feature_cols, top_n=5)

    if top_corrs:
        print("\nTop correlations (sample):")
        for col1, col2, corr in top_corrs:
            print(f" - {col1} to {col2}: {corr:.3f}")

    # ----- Multicollinearity -----

    print("\n" + "=" * 30)
    print("MULTICOLLINEARITY CHECK")
    print("=" * 30)

    print(f"\nNumber of features before removing multicollinearity: {len(feature_cols)}")

    # TODO this removes some of the interaction features we made, maybe make a feature priority list?
    # 2. Remove highly correlated features WITH VISUALIZATION
    clean_features, removed_features, high_corr_pairs = remove_multicollinearity(
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
    print(f"\nNumber of features after removing multicollinearity: {len(feature_cols)}")

    # ----- Scaling -----

    print("\n" + "=" * 30)
    print("SCALING FEATURES")
    print("=" * 30)

    _train, _test = train_test_split(engineered_train, test_size=0.25, random_state=42)
    excluded = ["user_id", "product_id", "order_id", "reordered"] + removed_features

    scaled_train, scalar = scale_features(
        df=_train,
        excluded_columns=excluded,
    )

    scaled_test, _ = scale_features(
        df=_test,
        excluded_columns=excluded,
        scalar=scalar,
    )

    # shut up pycharm again
    # noinspection PyPep8Naming
    X_train = scaled_train[feature_cols]
    y_train = scaled_train["reordered"]

    # shut up pycharm again v2
    # noinspection PyPep8Naming
    X_test = scaled_test[feature_cols]
    y_test = scaled_test["reordered"]

    # Show final features
    print(f"\nFinal features for modeling ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols[:15], 1):
        print(f"{i:2}. {col}")
    if len(feature_cols) > 15:
        print(f"... and {len(feature_cols) - 15} more")

    # ----- Modeling -----

    # print("=" * 30)
    # print("Task A")
    # print("=" * 30)
    #
    # print("Linear Training")
    # linear_c = linear_regressor(X_train, y_train)
    #
    # print("Lasso Training")
    # lasso_c = lasso_classifier(X_train, y_train, alpha=0.1)
    #
    # print("Ridge Training")
    # ridge_c = ridge_classifier(X_train, y_train, alpha=10)
    #
    # print("SVM-Linear Training")
    # svm_l_c = svm_linear_classifier(X_train, y_train, C=1.0, max_iter=10000)
    #
    #
    # print("K-NN Training")
    # knn_c = knn_classifier(X_train, y_train, n_neighbors=5)
    #
    # print("Decision Tree Training")
    # decision_tree_c = decision_tree_classifier(X_train, y_train, max_depth=15)
    #
    # print("Random Forest Training")
    # random_forest_c = random_forest_classifier(X_train, y_train, n_estimators=100, max_depth=10)
    #
    # # NVIDIA GPUs
    # print("XGBoost Training")
    # xgboost_c = xgboost_classifier(
    #     X_train,
    #     y_train,
    #     n_estimators=500,
    #     max_depth=10,
    #     learning_rate=0.1,
    #     device="cuda",
    # )

    # AMD GPUs
    # print("LightGBM Training")
    # lightgbm_c = lightgbm_classifier(
    #     X_train,
    #     y_train,
    #     n_estimators=100,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     device="gpu",
    # )

    print("LightGBM Training")
    lightgbm_c = lightgbm_classifier(
        X_train,
        y_train,
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        device="gpu",
    )

    # regression_models = [
    #     linear_c,
    #     lasso_c,
    #     ridge_c,
    # ]

    classification_models = [
        # svm_l_c
        # knn_c,
        # random_forest_c,
        # decision_tree_c,
        # xgboost_c,
        lightgbm_c,
    ]

    # for model in regression_models:
    #     evaluate_regressor(model, X_test, y_test, model.__class__.__name__)

    for model in classification_models:
        evaluate_classifier(model, X_test, y_test, model.__class__.__name__)

    print("=" * 30)
    print("Task B")
    print("=" * 30)

    print("Linear Training")
    linear_r = linear_regressor(X_train, y_train)

    print("Lasso Training")
    lasso_r = lasso_regressor(X_train, y_train, alpha=0.01)

    print("Ridge Training")
    ridge_r = ridge_regressor(X_train, y_train, alpha=10)

    print("ElasticNet Training")
    elastic_net_r = elastic_net_regressor(
        X_train,
        y_train,
        alpha=0.01,
        l1_ratio=0.5,
    )

    # print("SVM Linear Training")
    # svm_l_r = svm_linear_regressor(X_train, y_train, C=1.0)

    print("K-NN Training")
    knn_r = knn_regressor(
        X_train,
        y_train,
        n_neighbors=5,
    )

    print("Decision Tree Training")
    decision_tree_r = decision_tree_regressor(
        X_train,
        y_train,
        max_depth=10,
    )

    print("Random Forest Training")
    random_forest_r = random_forest_regressor(
        X_train,
        y_train,
        n_estimators=50,
    )

    print("XGBoost Training")
    xgboost_r = xgboost_regressor(
        X_train,
        y_train,
        device="cuda",
    )

    print("LightGBM Training")
    lightgbm_r = lightgbm_regressor(
        X_train,
        y_train,
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        device="gpu",
    )

    regression_models = [
        linear_r,
        lasso_r,
        ridge_r,
        elastic_net_r,
        # svm_l_r,
        knn_r,
        decision_tree_r,
        random_forest_r,
        xgboost_r,
    ]

    for model in regression_models:
        evaluate_regressor(model, X_test, y_test, model.__class__.__name__)

    print("\n" + "=" * 30)
    print("TASK B Accuracy")
    print("=" * 30)

    for model in regression_models:
        regression_accuracy(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            model.__class__.__name__,
        )


if __name__ == "__main__":
    main()
