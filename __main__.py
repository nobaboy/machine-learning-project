from pandas import DataFrame

from models import (
    logistic_classifier,
    lasso_classifier,
    ridge_classifier,
    knn_classifier,
    svm_linear_classifier,
    svm_kernel_classifier,
    decision_tree_classifier,
    random_forest_classifier,
    xgboost_classifier,
    lightgbm_classifier,

    linear_regressor,
    lasso_regressor,
    ridge_regressor,
    elastic_net_regressor,
    svm_linear_regressor,
    svm_kernel_regressor,
    knn_regressor,
    decision_tree_regressor,
    random_forest_regressor,
    xgboost_regressor,
    lightgbm_regressor,
)

from utils import get_feature_names, get_top_correlations
from utils.datasets import prep_task_a_dataset, prep_task_b_dataset
from utils.evaluation import evaluate_classifier, evaluate_regressor, evaluate_regressor_accuracy
from utils.feature import create_features
from utils.loader import load_data, calculate_memory_usage, format_memory_size, optimize_memory_usage
from utils.preprocessing import impute_column, remove_outliers, winsorize_outliers
from utils.visualization import visualize_memory_usage, analyze_and_visualize_missing


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

    engineered_train = create_features(
        prior=prior_orders,
        # orders=orders,
        train_pairs=train_with_labels[["user_id", "product_id", "reordered"]],
    )

    feature_cols = get_feature_names(engineered_train)

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

    # ----- Modeling -----

    print("\n" + "=" * 30)
    print("Task A (Classification)")
    print("=" * 30)

    # noinspection PyPep8Naming
    X_train, y_train, X_test, y_test = prep_task_a_dataset(
        engineered_train,
        feature_cols,
        corr_threshold=0.85,
        sample_size=5000,
        test_size=0.25,
        random_state=42,
    )

    print("Linear Training")
    log_c = logistic_classifier(X_train, y_train, max_iter=200, C=1.0)

    print("Lasso Training")
    lasso_c = lasso_classifier(X_train, y_train, alpha=0.1)

    print("Ridge Training")
    ridge_c = ridge_classifier(X_train, y_train, alpha=10)

    print("SVC-Linear Training")
    svm_l_c = svm_linear_classifier(X_train, y_train, C=1.0, max_iter=1000)

    print("SVC-RBF Training")
    svm_k_c = svm_kernel_classifier(X_train, y_train, C=1.0, kernel="rbf", max_iter=5000)

    print("K-NN Training")
    knn_c = knn_classifier(X_train, y_train, n_neighbors=9)

    print("Decision Tree Training")
    decision_tree_c = decision_tree_classifier(X_train, y_train, max_depth=10)

    print("Random Forest Training")
    random_forest_c = random_forest_classifier(X_train, y_train, n_estimators=100, max_depth=15)

    # NVIDIA GPUs
    print("XGBoost Training")
    xgboost_c = xgboost_classifier(
        X_train,
        y_train,
        n_estimators=500,
        max_depth=10,
        learning_rate=0.1,
        device="cuda",
    )

    # AMD GPUs
    # print("LightGBM Training")
    # lightgbm_c = lightgbm_classifier(
    #     X_train,
    #     y_train,
    #     n_estimators=300,
    #     max_depth=10,
    #     learning_rate=0.05,
    #     device="gpu",
    # )

    classification_models = [
        log_c,
        lasso_c,
        ridge_c,
        svm_l_c,
        svm_k_c,
        knn_c,
        decision_tree_c,
        random_forest_c,
        xgboost_c,
        # lightgbm_c,
    ]

    for model in classification_models:
        evaluate_classifier(model, X_test, y_test, model.__class__.__name__)

    print("=" * 30)
    print("Task B (Regression): days_since_prior_order")
    print("=" * 30)

    # noinspection PyPep8Naming
    X_train, y_train, X_test, y_test = prep_task_b_dataset(
        prior,
        orders,
        test_size=0.25,
        random_state=42,
    )

    print("Linear Training")
    linear_r = linear_regressor(X_train, y_train)

    print("Lasso Training")
    lasso_r = lasso_regressor(X_train, y_train, alpha=0.01)

    print("Ridge Training")
    ridge_r = ridge_regressor(X_train, y_train, alpha=10)

    print("ElasticNet Training")
    elastic_net_r = elastic_net_regressor(X_train, y_train, alpha=0.01, l1_ratio=0.5)

    print("SVR-Linear Training")
    svm_l_r = svm_linear_regressor(X_train, y_train, C=1.0)

    print("SVR-RBF Training")
    svm_k_r = svm_kernel_regressor(X_train, y_train, C=1.0, kernel="rbf", max_iter=5000)

    print("K-NN Training")
    knn_r = knn_regressor(X_train, y_train, n_neighbors=9)

    print("Decision Tree Training")
    decision_tree_r = decision_tree_regressor(X_train, y_train, max_depth=10)

    print("Random Forest Training")
    random_forest_r = random_forest_regressor(X_train, y_train, n_estimators=150, max_depth=15)

    # NVIDIA GPUs
    print("XGBoost Training")
    xgboost_r = xgboost_regressor(
        X_train,
        y_train,
        device="cuda",
    )

    # AMD GPUs
    # print("LightGBM Training")
    # lightgbm_r = lightgbm_regressor(
    #     X_train,
    #     y_train,
    #     n_estimators=300,
    #     max_depth=-1,
    #     learning_rate=0.05,
    #     device="gpu",
    # )

    regression_models = [
        linear_r,
        lasso_r,
        ridge_r,
        elastic_net_r,
        svm_l_r,
        svm_k_r,
        knn_r,
        decision_tree_r,
        random_forest_r,
        xgboost_r,
        # lightgbm_r,
    ]

    for model in regression_models:
        evaluate_regressor(model, X_test, y_test, model.__class__.__name__)

    print("\n" + "=" * 30)
    print("TASK B Accuracy")
    print("=" * 30)

    for model in regression_models:
        train_accuracy = evaluate_regressor_accuracy(model, X_train, y_train)
        test_accuracy = evaluate_regressor_accuracy(model, X_test, y_test)

        print(f"\n{model.__class__.__name__}")
        print(f"Train: {train_accuracy:.3f}")
        print(f"Test: {test_accuracy:.3f}")


if __name__ == "__main__":
    main()
