import numpy as np
from pandas import DataFrame

__all__ = (
    "create_features",
    "build_user_features",
)


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

    out = user_core.merge(user_basket, on=["user_id"], how="left")

    if "days_since_prior_order" in prior.columns:
        temp = prior[["user_id", "order_id", "days_since_prior_order"]].drop_duplicates()
        temp["pos_dspo"] = temp["days_since_prior_order"].where(temp["days_since_prior_order"] > 0)

        mean_gap = temp.groupby("user_id")["pos_dspo"].mean()
        last_gap = temp.groupby("user_id")["days_since_prior_order"].last()

        out["mean_days_between_orders"] = out["user_id"].map(mean_gap).fillna(0)
        out["last_days_since_prior"] = out["user_id"].map(last_gap).fillna(0)

    return out


def build_product_features(prior: DataFrame) -> DataFrame:
    base = prior.groupby("product_id").agg(
        product_orders=("order_id", "nunique"),
        product_users=("user_id", "nunique"),
        product_reorder_rate=("reordered", "mean"),
        product_avg_cart_position=("add_to_cart_order", "mean"),
    ).reset_index()

    if "order_number" in prior.columns:
        temp = prior[["product_id", "order_number"]].dropna()

        prod_time = temp.groupby("product_id").agg(
            product_avg_order_number=("order_number", "mean"),
            product_max_order_number=("order_number", "max"),
        ).reset_index()

        prod_time["product_trend_rate"] = (
            prod_time["product_avg_order_number"] / prod_time["product_max_order_number"]
        ).fillna(0)

        base = base.merge(prod_time, on=["product_id"], how="left")

    return base


def build_user_product_features(prior: DataFrame) -> DataFrame:
    agg = {
        "up_purchase_count": ("order_id", "count"),
        "up_avg_reorder_rate": ("reordered", "mean"),
    }

    if "order_number" in prior.columns:
        agg["up_last_order_number"] = ("order_number", "max")

    up = prior.groupby(["user_id", "product_id"]).agg(**agg).reset_index()

    if "order_number" in prior.columns:
        user_last_order_number = prior.groupby("user_id").agg(
            user_last_order_number=("order_number", "max"),
        ).reset_index()

        up = up.merge(user_last_order_number, on=["user_id"], how="left")

        up["up_orders_since_last_purchase"] = (
            up["user_last_order_number"] - up["up_last_order_number"]
        ).fillna(0)

        up = up.drop(columns=["user_last_order_number"])

    return up


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
