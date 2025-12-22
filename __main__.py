import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, calculate_memory_usage, format_memory_size, optimize_memory_usage, imputerColumn , remove_outliers
from visualization import visualize_memory_usage, analyze_and_visualize_missing

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer , SimpleImputer

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
    print(f"\nReduced total memory usage from {format_memory_size(mem_usage_before)} to {format_memory_size(mem_usage_after)} ({diff:.1f}% reduction)")

    # ----- Visualization -----

    visualize_memory_usage(mem_usage_before, mem_usage_after)

    aisles, departments, products = datasets["aisles"], datasets["departments"], datasets["products"]
    orders, prior, train = datasets["orders"], datasets["prior"], datasets["train"]

    print("\nMerging datasets...")

    # prior_orders = prior.merge(
    #     orders,
    #     on="order_id"
    # # )
    #
    #
    # products_full = (
        # products
            # .merge(aisles, on="aisle_id")
            # .merge(departments, on="department_id"
    # )
    # )

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
    ) # final version


    missing_cols = analyze_and_visualize_missing(data_full)

    base_features = data_full[["user_id", "product_id"]].drop_duplicates()

    train_labels = train[["order_id", "product_id", "reordered"]]
    train_user = train_labels.merge(orders[["order_id", "user_id"]], on="order_id")

    train = base_features.merge(
        train_user[["user_id", "product_id", "reordered"]],
        on=["user_id", "product_id"],
        how="left", # keep all the pairs from prior
    )

    # keeping all the pairs from prior introduces NaN values, which in python is a floating point number
    # so fill those missing values with 0 and optimize the dtype again
    train["reordered"] = train["reordered"].fillna(0).astype("int8")

    # ----- Imputation -----

    # TODO make this a for loop (even though it's only 1 column with missing values), or even better, replace it
    #      with the preprocessing pipeline and provide missing_cols in the builder function

    data_full = imputerColumn(data_full, "days_since_prior_order", "median") # we choose to remove Outliers over Treatment because we have a big amount of data so DELETE them don't make any damage

    #  ----- Outlier Handling -----

    columns_to_check = ['add_to_cart_order', 'days_since_prior_order']
    # TODO the professor suggested winsorizing, check that out instead of removing outliers
    data_full = remove_outliers(data_full, columns_to_check)
    # ----- Misc. (just some info at the moment) -----

    print(train.info())
    print("=" * 50)
    print(data_full.info())
    print(format_memory_size(calculate_memory_usage(train)))
    print(format_memory_size(calculate_memory_usage(data_full)))



if __name__ == "__main__":
    main()
