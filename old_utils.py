from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.visualization import plot_correlation_heatmap


def remove_outliers(df: DataFrame, cols: list[str]):
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        print(f"\nProcessing: {col}")

        # Statistics
        before_count = len(df)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Define what to keep and what to remove
            mask_keep = (df[col] >= lower) & (df[col] <= upper)
            outliers_count = (~mask_keep).sum()  # Count the inverse True outliers

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(6, 6))

            # Before plot
            ax1.hist(df[col], bins=30, alpha=0.6, color='red', label='Outliers included')
            ax1.set_title(f'BEFORE: {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Frequency')
            ax1.legend()

            # After plot
            ax2.hist(df[mask_keep][col], bins=30, alpha=0.6, color='green', label='Outliers removed')
            ax2.set_title(f'AFTER: {col}')
            ax2.set_xlabel(col)
            ax2.set_ylabel('Frequency')
            ax2.legend()

            plt.suptitle(
                f'Outlier Removal: {col}\nRemoved {outliers_count} outliers ({outliers_count / before_count * 100:.1f}%)')
            plt.show()

            # Apply filter
            df = df[mask_keep].copy()

            print(f"   Removed: {outliers_count} outliers ({outliers_count / before_count * 100:.1f}%)")

    return df


# FIXME doesn't scale engineered features
def feature_scaling(
    df: DataFrame,
    columns_to_exclude: list[str],
    method: Literal["standard", "minmax"] = "minmax",
):
    # Select numeric columns
    numeric_cols = df.select_dtypes(
        include=['int16', 'int32', 'int8', 'float16', 'float32', 'float64']
    ).columns

    # Filter out excluded columns
    columns_to_scale = [col for col in numeric_cols if col not in columns_to_exclude]

    # TODO when u see this comment : there's an logic
    if len(columns_to_scale) == 0:
        print("No columns to scale Check your data types or exclusion list")
        return df, None

    print(f"\nScaling {len(columns_to_scale)} numeric columns: ")
    print(f"Columns to scale: {columns_to_scale}")

    scaler = None # We Initialize SCALER here for the else
    # Choose scaler based on method
    if method == 'standard':
        scaler = StandardScaler() # I'm using Logistic Regression and SVM work better with StandardScaler
    elif method == 'minmax':
        scaler = MinMaxScaler() # K-Nearest Neighbors KNN - distance based
    else:
        print("method must be 'standard' or 'minmax'")

    # Apply scaling
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df, scaler


def featuresEng(prior_df, orders_df, train_df):
    print("Creating complete features from PRIOR data: ")

    # 1. USER Feature (from prior)
    print("1. User features:")
    user_features = prior_df.groupby('user_id').agg(
        user_total_orders=('order_id', 'nunique'),
        user_total_items=('product_id', 'count'),
        user_reorder_rate=('reordered', 'mean')
    ).reset_index()

    # 2. Prodect Feature (from prior)
    print("2. Product features: ")
    product_features = prior_df.groupby('product_id').agg(
        product_orders=('order_id', 'nunique'),
        product_users=('user_id', 'nunique')
    ).reset_index()

    # 3. User-Product Feature from prior
    print("3. User-product features: ")
    up_features = prior_df.groupby(['user_id', 'product_id']).agg(
        up_count=('order_id', 'count'),
        up_last_order=('order_number', 'max')
    ).reset_index()

    # 4. Time Feature from orders
    print("4. Time features: ")
    # First get order_id for train data
    train_with_order_id = train_df.copy()

    time_features = orders_df[['order_id', 'order_dow', 'order_hour_of_day']].copy()
    time_features.columns = ['order_id', 'order_day', 'order_hour']

    # 5. Merge all with Train
    print("5. Merging with train...")
    features = train_with_order_id.copy()

    # Merge each feature set
    features = features.merge(user_features, on='user_id', how='left')
    features = features.merge(product_features, on='product_id', how='left')
    features = features.merge(up_features, on=['user_id', 'product_id'], how='left')

    # Only merge time features if we have order_id
    if 'order_id' in features.columns:
        features = features.merge(time_features, on='order_id', how='left')

    # 7. Add Non-Linear Features
    print("7. Adding non-linear features: ")

    features['orders_x_count'] = features['user_total_orders'] * features['up_count']
    features['log_up_count'] = np.log1p(features['up_count'])

    # Add one more interaction feature
    features['user_orders_x_product_orders'] = features['user_total_orders'] * features['product_orders']

    print(f" Created {features.shape[1] - 4} features total")
    print(f" Shape: {features.shape}")

    return features


def get_feature_names(df):
    exclude_cols = ['user_id', 'product_id', 'order_id', 'reordered']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\n Feature columns ({len(feature_cols)} total):")
    for col in feature_cols:
        print(f"  - {col}")

    return feature_cols


def multicollinearity(
    df: DataFrame,
    feature_cols: list,
    corr_threshold: float = 0.85,
    sample_size: int = 10000,
    visualize: bool = True,
):
    print(f"Checking multicollinearity for {len(feature_cols)} features ")

    # Calculate correlation matrix
    print("Calculating correlation matrix: ")
    corr_matrix = df[feature_cols].corr().abs()

    # Create visualization BEFORE removing features
    if visualize and len(feature_cols) > 1:
        plot_correlation_heatmap(corr_matrix, title="Feature Correlation Heatmap (Before Remove)")

    # Find highly correlated pairs
    high_corr_pairs = []
    features_to_remove = set()

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr = corr_matrix.iloc[i, j]

            if corr > corr_threshold:
                col1 = feature_cols[i]
                col2 = feature_cols[j]
                high_corr_pairs.append((col1, col2, corr))

                # Decide which to remove keep the one with higher variance
                var1 = df[col1].var()
                var2 = df[col2].var()

                if var1 >= var2:
                    features_to_remove.add(col2)
                    print(f"   Removing {col2} (kept {col1}, corr={corr:.3f})")
                else:
                    features_to_remove.add(col1)
                    print(f"   Removing {col1} (kept {col2}, corr={corr:.3f})")

    # Results
    features_to_keep = [col for col in feature_cols if col not in features_to_remove]

    # Create visualization AFTER removing features
    if visualize and len(features_to_keep) > 1 and len(features_to_keep) <= 20:
        remaining_corr = df[features_to_keep].corr().abs()
        plot_correlation_heatmap(remaining_corr,
                                 title="Feature Correlation Heatmap After Remove",
                                 figsize=(10, 8))

    if features_to_remove:
        print(f"\n Removed features: {list(features_to_remove)}")

    return features_to_keep, list(features_to_remove), high_corr_pairs


def get_top_correlations(
    df: DataFrame,
    feature_cols: list,
    top_n: int = 10,
    sample_size: int = 5000,
):
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    corr_matrix = df[feature_cols].corr().abs()

    # Get top correlations
    correlations = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr = corr_matrix.iloc[i, j]
            if corr > 0.1:  # Ignore very low correlations
                correlations.append({
                    'feature1': feature_cols[i],
                    'feature2': feature_cols[j],
                    'correlation': corr
                })

    correlations.sort(key=lambda x: x['correlation'], reverse=True)
    return correlations[:top_n]
