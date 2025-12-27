import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer , SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
__all__ = (
    "load_data",
    "calculate_memory_usage",
    "format_memory_size",
    "optimize_memory_usage",
)

def load_data(path: str) -> DataFrame:
    df = pd.read_csv(path, engine="pyarrow")
    mem_usage = calculate_memory_usage(df)
    print(f"Loaded '{path}' ({format_memory_size(mem_usage)})")
    return df


def calculate_memory_usage(*dfs: DataFrame) -> int:
    return sum(df.memory_usage().sum() for df in dfs)


# TODO move out sizes into consts
def format_memory_size(size: int) -> str:
    if size < 1024:
        return f"{size:.0f} B"
    if size < 1024 ** 2:
        return f"{size / 1024:.2f} KiB"
    return f"{size / 1024 ** 2:.2f} MiB"


def optimize_memory_usage(name: str, df: DataFrame):
    before = calculate_memory_usage(df)

    for col in df.columns:
        col_dtype = df[col].dtype

        if pd.api.types.is_integer_dtype(col_dtype):
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif pd.api.types.is_float_dtype(col_dtype):
            df[col] = df[col].astype(np.float32)

        elif col_dtype == object:
            unique = len(df[col].unique())
            total = len(df[col])

            # we only want to convert high cardinality columns (e.g. eval_set from orders dataset)
            if unique / total < 0.5:
                df[col] = df[col].astype("category")

    after = calculate_memory_usage(df)
    diff = 100 * (before - after) / before
    print(f"Reduced memory usage of '{name}' from {format_memory_size(before)} to {format_memory_size(after)} ({diff:.1f}% reduction)")


def imputerColumn(data: DataFrame, column: str, strategy: str, **kwargs):
    if column not in data.columns:
        print(f"Column {column} not found")
        return data

    print(f"Imputing column {column} with strategy {strategy}")

    # Create figure for before/after
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Column: {column} - Strategy: {strategy}', fontsize=14, fontweight='bold')

    # Plot BEFORE imputation
    axes[0].set_title(f'Before Imputation: {data[column].isna().sum()}')

    # For numeric columns
    if np.issubdtype(data[column].dtype, np.number):
        axes[0].hist(data[column], bins=30, edgecolor='black', alpha=0.7, color='blue')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')


    missing_before = data[column].isna().sum()
    print(f"Missing values before: {missing_before}")

    # --------Simple strategies---------
    if strategy in ["median", "mean", "most_frequent"]:
        imputer = SimpleImputer(strategy=strategy)
        data[[column]] = imputer.fit_transform(data[[column]])

    # -------- Sentinel-------
    elif strategy == "sentinel":
        if np.issubdtype(data[column].dtype, np.integer):
            sentinel = kwargs.get("fill_value", -999)
        elif np.issubdtype(data[column].dtype, np.floating):
            sentinel = kwargs.get("fill_value", -999.0)
        else:
            sentinel = kwargs.get("fill_value", "MISSING")
        data[column] = data[column].fillna(sentinel)

    # --------Iterative Model-Based--------
    elif strategy == "iterative":
        numCol = data.select_dtypes(include=[np.number]).columns

        iterative_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 50),
                random_state=42
            ),
            max_iter=kwargs.get("max_iter", 10),
            random_state=42
        )
        data[numCol] = iterative_imputer.fit_transform(data[numCol])

    else:
        print(f"Unknown strategy: {strategy}")
        plt.close(fig)  # Close the figure if strategy is unknown
        return data

    # Plot AFTER imputation
    axes[1].set_title(f'After Imputation\nMissing: {data[column].isna().sum()}')

    # For numeric columns
    if np.issubdtype(data[column].dtype, np.number):
        axes[1].hist(data[column], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')

        # Add vertical lines for statistics (only if not sentinel with extreme values)
        if strategy != "sentinel":
            if strategy == "median":
                median_val = data[column].median()
                axes[1].axvline(median_val, color='red', linestyle='--',
                                label=f'Median: {median_val:.2f}')
            elif strategy == "mean":
                mean_val = data[column].mean()
                axes[1].axvline(mean_val, color='red', linestyle='--',
                                label=f'Mean: {mean_val:.2f}')
            axes[1].legend()

    # For categorical columns
    else:
        value_counts = data[column].value_counts().head(10)
        bars = axes[1].bar(range(len(value_counts)), value_counts.values,
                           color='green', alpha=0.7)
        axes[1].set_xticks(range(len(value_counts)))
        axes[1].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[1].set_ylabel('Count')

        # Add percentage labels
        total = value_counts.sum()
        for i, (bar, val) in enumerate(zip(bars, value_counts.values)):
            percentage = 100 * val / total
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)

    # Adjust layout and show plot
    plt.show()

    print(f"Missing values after: {data[column].isna().sum()}")

    return data


def remove_outliers(df: DataFrame, cols: list[str]):
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        print(f"\n Processing: {col}")

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
            print(f"   Bounds: [{lower:.2f}, {upper:.2f}]")

    return df


from sklearn.preprocessing import StandardScaler, MinMaxScaler


def feature_scaling(data: DataFrame, method: str, columns_to_exclude: list[str]):
    # Select numeric columns
    numeric_cols = data.select_dtypes(
        include=['int16', 'int32', 'int8', 'float16', 'float32', 'float64']
    ).columns

    # Filter out excluded columns
    columns_to_scale = [col for col in numeric_cols if col not in columns_to_exclude]

    # TODO when u see this comment : there's an logic
    if len(columns_to_scale) == 0:
        print("No columns to scale Check your data types or exclusion list")
        return data, None

    print(f"\nScaling {len(columns_to_scale)} numeric columns: ")
    print(f"Columns to scale: {columns_to_scale}")

    # Choose scaler based on method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        print("method must be 'standard' or 'minmax'")

    # Apply scaling
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return data, scaler

def create_simple_features(data_full, train_data, orders_data):

    print("Step 1: Creating user features: ")
    # User-level features
    user_features = data_full.groupby('user_id').agg(
        user_total_orders=('order_id', 'nunique'),  # How many orders total
        user_avg_items=('product_id', 'count'),  # Average items per user
        user_reorder_rate=('reordered', 'mean')  # User's reorder rate
    ).reset_index()

    print("Step 2: Creating product features: ")
    # Product-level features
    product_features = data_full.groupby('product_id').agg(
        product_popularity=('order_id', 'nunique'),  # How many times ordered
        product_avg_cart_pos=('add_to_cart_order', 'mean')  # Average position in cart
    ).reset_index()
    print("Step 3: Creating user-product features: ")

    # User-Product interaction features
    up_features = data_full.groupby(['user_id', 'product_id']).agg( # It lets you calculate multiple statistics at once
        up_times_bought=('order_id', 'count'),  # How many times user bought this product
        up_last_order_num=('order_number', 'max')  # Last order when bought
    ).reset_index()
    print("Step 4: Adding time features: ")

    # Time features
    time_features = orders_data[['order_id', 'order_dow', 'order_hour_of_day']].copy()
    time_features.columns = ['order_id', 'order_day', 'order_hour']
    print("Step 5: Merging all features: ")

    # Start with train data
    features = train_data.copy()

    # Merge user features
    features = features.merge(user_features, on='user_id', how='left')

    # Merge product features
    features = features.merge(product_features, on='product_id', how='left')

    # Merge user-product features
    features = features.merge(up_features, on=['user_id', 'product_id'], how='left')

    # Merge time features
    if 'order_id' in features.columns:
        features = features.merge(time_features, on='order_id', how='left')

    # Add one simple interaction feature
    features['orders_x_bought'] = features['user_total_orders'] * features['up_times_bought']

    print(f" Created {features.shape[1]} features total")
    print(f" Final shape: {features.shape}")

    return features


def get_feature_columns(features_df):

    # Columns to exclude
    exclude_cols = ['user_id', 'product_id', 'order_id', 'reordered']

    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    print(f"\n Feature columns ({len(feature_cols)} total):")
    for col in feature_cols:
        print(f"  - {col}")

    return feature_cols


