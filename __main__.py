import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix

from utils import load_data, calculate_memory_usage, format_memory_usage, optimize_memory_usage


def imputer_median(data, colName):
    if colName in data.columns:
        imputer = SimpleImputer(strategy="median")
        data[[colName]] = imputer.fit_transform(data[[colName]])
    return data  # Added return statement

def remove_putliers(data, col):
    if isinstance(col, str):
        col = [col]  # Convert single column name to list

    for cl in col:
        if cl in Data.columns and Data[cl].dtype in ["int64", "float64", "int32", "float32"]:
            Q1 = Data[cl].quantile(0.25)
            Q3 = Data[cl].quantile(0.75)
            IQT = Q3 - Q1

            # Avoid division by zero
            if IQT > 0:
                lower = Q1 - 1.5 * IQT
                upper = Q3 + 1.5 * IQT

                Data = Data[(Data[cl] >= lower) & (Data[cl] <= upper)]

    return Data


def main():
    datasets: dict[str, DataFrame] = {
        "aisles": load_data("data/aisles.csv"),
        "departments": load_data("data/departments.csv"),
        "products": load_data("data/products.csv"),
        "orders": load_data("data/orders.csv"),
        "prior": load_data("data/order_products__prior.csv"),
        "train": load_data("data/order_products__train.csv"),
    }

    mem_usage_before = calculate_memory_usage(*datasets.values())
    
    for name, dataset in dataset.items():
        optimize_memory_usage(name, dataset)

    mem_usage_after = calculate_memory_usage(*datasets.values())
    diff = 100 * (mem_usage_before - mem_usage_after) / mem_usage_before
    print(f"\n\nReduced total memory usage from {format_memory_size(mem_usage_before)} to {format_memory_size(mem_usage_after)} ({diff:.1f}% reduction)\n")

    aisles, departments, products = datasets["aisles"], datasets["departments"], datasets["products"]
    orders, prior, train = datasets["orders"], datasets["prior"], datasets["train"]

    # 1. Merge PRIOR data
    prior_orders = (
        prior.merge(orders, on="order_id")
        # .merge(products, on="product_id")
        # .merge(aisles, on="aisle_id")
        # .merge(departments, on="department_id")
    )

    # Preparing labels from TRAIN (NO DATA LEAKAGE)

    # 2. Get labels from TRAIN
    train_labels = train[["order_id", "product_id", "reordered"]]

    train_with_user = train_labels.merge(
        orders[["order_id", "user_id"]],  # Only take these two columns
        on="order_id",
    )

    # 3. Create base user-product
    base_features = prior_orders[["user_id", "product_id"]].drop_duplicates()

    # 4. FINAL MERGE Combine features with labels
    train_data = base_features.merge(
        train_with_user[["user_id", "product_id", "reordered"]],
        on=["user_id", "product_id"],
        how="left"  # Keep all pairs from prior
    )

    # Fill NaN with 0 (products not in last order)
    train_data["reordered"] = train_data["reordered"].fillna(0)

    # Handle missing values in orders products
    print(prior_orders.isnull().sum())
    # Check for missing values
    missing_info = prior_orders.isna().sum()
    missing_cols = missing_info[missing_info > 0]

    if not missing_cols.empty:
        for col, count in missing_cols.items():
            percent = (count / len(prior_orders)) * 100

        if "days_since_prior_order" in missing_cols.index:
            median_val = prior_orders["days_since_prior_order"].median()
            prior_orders["days_since_prior_order"] = prior_orders["days_since_prior_order"].fillna(median_val)

    print(prior_orders.isnull().sum())
    
    # Remove outliers
    columns_to_check = ["add_to_cart_order", "days_since_prior_order"]

    prior_orders = OutliersRemover(prior_orders, columns_to_check)

    print(train_data.info())
    print("=" * 50)
    print(prior_orders.info())


if __name__ == "__main__":
    main()
    