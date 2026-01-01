from pandas import DataFrame
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils import load_data, calculate_memory_usage, format_memory_size, optimize_memory_usage, imputerColumn, \
    remove_outliers, featuresEng, get_feature_names, feature_scaling, multicollinearity, get_top_correlations
from visualization import visualize_memory_usage, analyze_and_visualize_missing

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

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
    diff = 100 * (mem_usage_before - mem_usage_after) / mem_usage_before
    print(
        f"\nReduced total memory usage from {format_memory_size(mem_usage_before)} to {format_memory_size(mem_usage_after)} ({diff:.1f}% reduction)")

    # ----- Visualization -----
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

    missing_cols = analyze_and_visualize_missing(data_full)

    base_features = data_full[["user_id", "product_id"]].drop_duplicates()

    train_labels = train[["order_id", "product_id", "reordered"]]
    train_user = train_labels.merge(orders[["order_id", "user_id"]], on="order_id")

    train = base_features.merge(
        train_user[["user_id", "product_id", "reordered"]],
        on=["user_id", "product_id"],
        how="left",  # keep all the pairs from prior
    )

    # keeping all the pairs from prior introduces NaN values, which in python is a floating point number
    # so fill those missing values with 0 and optimize the dtype again
    train["reordered"] = train["reordered"].fillna(0).astype("int8")

    # ----- Imputation -----
    data_full = imputerColumn(data_full, "days_since_prior_order", "median")

    #  ----- Outlier Handling -----
    columns_to_check = ['days_since_prior_order', 'add_to_cart_order']
    data_full = remove_outliers(data_full, columns_to_check)

    # ----- Feature Engineering -----
    print("\nFEATURE ENGINEERING")

    # Extract just prior data
    prior_only = prior.merge(
        orders[['order_id', 'user_id', 'order_number']],
        on='order_id'
    )

    # Create complete features
    engineered_train = featuresEng(
        prior_df=prior_only,
        orders_df=orders,
        train_df=train
    )

    # Get ALL feature names
    feature_cols = get_feature_names(engineered_train)

    print(f"\nOriginal features ({len(feature_cols)} total):")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"{i:2}. {col}")
    if len(feature_cols) > 10:
        print(f"... and {len(feature_cols) - 10} more")

    # ---- MULTICOLLINEARITY CHECK ----
    print("MULTICOLLINEARITY CHECK")

    # 1. Show top correlations
    top_corrs = get_top_correlations(engineered_train, feature_cols, top_n=5)
    print("\nTop 5 feature correlations:")
    for col1, col2, corr in top_corrs:
        print(f"  {col1} - {col2}: {corr:.3f}")

    # 2. Remove highly correlated features
    clean_features, removed_features, high_corr_pairs = multicollinearity(
        df=engineered_train,
        feature_cols=feature_cols,
        corr_threshold=0.85,
        sample_size=5000,
    )

    # Update feature list
    feature_cols = clean_features
    print(f"\nFeatures after removing multicollinearity: {len(feature_cols)}")

    # ----- Scale the features -----
    print("SCALING FEATURES")

    # Split
    train_df, test_df = train_test_split(engineered_train, test_size=0.25, random_state=42)

    # Scale train
    train_df_scaled, scaler = feature_scaling(
        train_df, feature_cols, method='standard', fit=True
    )

    # Scale test
    test_df_scaled, _ = feature_scaling(
        test_df, feature_cols, method='standard', scaler=scaler, fit=False
    )

    # Prepare X and y for modeling
    X_train = train_df_scaled[feature_cols]
    y_train = train_df_scaled['reordered']

    X_test = test_df_scaled[feature_cols]
    y_test = test_df_scaled['reordered']

    print(f"\nData prepared:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Show final features
    print(f"\nFinal features for modeling ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols[:15], 1):  # Show first 15
        print(f"{i:2}. {col}")
    if len(feature_cols) > 15:
        print(f"... and {len(feature_cols) - 15} more")

    # ---------------- Train Models ----------------
    linear = LinearRegression()
    ridge = Ridge(alpha=10)
    lasso = Lasso(alpha=0.01, max_iter=10000)

    linear.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    print("\nModels trained: Linear, Ridge (alpha=10), Lasso (alpha=0.01)")

    # Predictions on TRAIN data
    y_pred_linear = np.ravel(linear.predict(X_train))
    y_pred_ridge = np.ravel(ridge.predict(X_train))
    y_pred_lasso = np.ravel(lasso.predict(X_train))

    # ---------------- Coefficient Analysis ----------------
    print("\n" + "=" * 60)
    print("COEFFICIENT ANALYSIS")
    print("=" * 60)

    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"Non-zero coefficients:")
    print(f"  Linear: {np.sum(linear.coef_ != 0)}")
    print(f"  Ridge:  {np.sum(ridge.coef_ != 0)}")
    print(f"  Lasso:  {np.sum(lasso.coef_ != 0)}")

    print(f"\nPercentage of non-zero coefficients:")
    print(f"  Linear: {np.sum(linear.coef_ != 0) / len(feature_cols) * 100:.1f}%")
    print(f"  Ridge:  {np.sum(ridge.coef_ != 0) / len(feature_cols) * 100:.1f}%")
    print(f"  Lasso:  {np.sum(lasso.coef_ != 0) / len(feature_cols) * 100:.1f}%")

    # ---------------- Summary ----------------
    print("SUMMARY")

    train_r2 = [r2_score(y_train, y_pred_linear),
                r2_score(y_train, y_pred_ridge),
                r2_score(y_train, y_pred_lasso)]

    models = ['Linear', 'Ridge', 'Lasso']
    best_train_idx = np.argmax(train_r2)

    print(f"\nBest model on training data: {models[best_train_idx]} (R² = {train_r2[best_train_idx]:.4f})")

    # KNN CLASSIFIER

    print("KNN CLASSIFIER")

    # ----- Train KNN with different k values -----

    k_values = 9

    # ----- Train final model with best k -----
    print(f"\nTraining final KNN model with k={k_values}...")
    final_knn = KNeighborsClassifier(n_neighbors=k_values)
    final_knn.fit(X_train, y_train)

    # Get predictions
    y_pred_train = final_knn.predict(X_train)
    y_pred_test = final_knn.predict(X_test)

    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)


    print("XGBOOST CLASSIFIER WITH GPU")

    # ----- Train XGBoost -----
    print("\nTraining XGBoost with GPU...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        device='cuda'
    )

    xgb_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_test_xgb = xgb_model.predict(X_test)
    test_acc_xgb = accuracy_score(y_test, y_pred_test_xgb)

    print(f"XGBoost Test Accuracy: {test_acc_xgb:.4f}")

if __name__ == "__main__":
    main()

