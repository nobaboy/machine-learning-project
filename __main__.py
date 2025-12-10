# import numpy as np
import pandas as pd
# import matplotlib as plt

# from pandas import DataFrame


def load_data(file_name: str):
    return pd.read_csv(file_name)


if __name__ == "__main__":
    # aisles = load_data("data/aisles.csv")
    # departments = load_data("data/departments.csv")

    orders = load_data("data/orders.csv")
    op_prior = load_data("data/order_products__prior.csv")

    merged_op = pd.merge(orders, op_prior, on="order_id")
