import numpy as np
from pandas import DataFrame

__all__ = ("create_features",)


# TODO mean days between orders, last order recency
def build_user_features(prior: DataFrame) -> DataFrame:
    user_core = prior.groupby("user_id").agg(
        user_total_orders=("order_id", "nunique"),
        user_total_items=("product_id", "count"),
        user_reorder_rate=("reordered", "mean"),
    ).reset_index()

    order_sizes = prior.groupby(["user_id", "order_id"]).size().rename("basket_size").reset_index()
    user_basket = order_sizes.groupby("user_id").agg(
        user_avg_basket_size=("basket_size", "mean"),
    ).reset_index()

    return user_core.merge(user_basket, on="user_id", how="left")


# TODO popularity over time
def build_product_features(prior: DataFrame) -> DataFrame:
    return prior.groupby("product_id").agg(
        product_orders=("order_id", "nunique"),
        product_users=("user_id", "nunique"),
        product_reorder_rate=("reordered", "mean"),
        product_avg_cart_position=("add_to_cart_order", "mean"),
    ).reset_index()


# TODO days since last purchase
def build_user_product_features(prior: DataFrame) -> DataFrame:
    return prior.groupby(["user_id", "product_id"]).agg(
        up_purchase_count=("order_id", "count"),
        # up_last_order_number=("order_number", "max"),
        up_avg_reorder_rate = ("reordered", "mean"),
    ).reset_index()


# TODO
def build_temporal_features(orders: DataFrame) -> DataFrame:
    ...


def create_features(prior: DataFrame, train_pairs: DataFrame) -> DataFrame:
    features = train_pairs.copy()

    user_features = build_user_features(prior)
    product_features = build_product_features(prior)
    up_features = build_user_product_features(prior)

    features = features.merge(user_features, on="user_id", how="left")
    features = features.merge(product_features, on="product_id", how="left")
    features = features.merge(up_features, on=["user_id", "product_id"], how="left")
    # features = features.merge(time_features, on="user_id", how="left")

    features["log_up_purchase_count"] = np.log1p(features["up_purchase_count"])
    features["user_orders_x_up_count"] = features["user_total_orders"] * features["up_purchase_count"]

    features["user_orders_x_product_orders"] = features["user_total_orders"] * features["product_orders"]

    return features
