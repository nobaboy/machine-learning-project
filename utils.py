import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = (
    "load_data",
    "calculate_memory_usage",
    "format_memory_size",
    "optimize_memory_usage",
    "imputerColumn",
    "remove_outliers",
    "featuresEng",
    "get_feature_names",
    "feature_scaling",
    "get_top_correlations",
    "multicollinearity"
)

# ---------------------------- Basic Utilities ----------------------------

def load_data(path: str) -> DataFrame:
    df = pd.read_csv(path, engine="pyarrow")
    mem_usage = calculate_memory_usage(df)
    print(f"Loaded '{path}' ({format_memory_size(mem_usage)})")
    return df

def calculate_memory_usage(*dfs: DataFrame) -> int:
    return sum(df.memory_usage().sum() for df in dfs)

def format_memory_size(size: int) -> str:
    if size < 1024:
        return f"{size:.0f} B"
    if size < 1024**2:
        return f"{size/1024:.2f} KiB"
    return f"{size/1024**2:.2f} MiB"

def optimize_memory_usage(name: str, df: DataFrame):
    before = calculate_memory_usage(df)
    for col in df.columns:
        col_dtype = df[col].dtype

        if pd.api.types.is_integer_dtype(col_dtype):
            col_min, col_max = df[col].min(), df[col].max()
            if np.iinfo(np.int8).min <= col_min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif np.iinfo(np.int16).min <= col_min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif np.iinfo(np.int32).min <= col_min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif pd.api.types.is_float_dtype(col_dtype):
            df[col] = df[col].astype(np.float32)

        elif col_dtype == object:
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype("category")

    after = calculate_memory_usage(df)
    print(f"Reduced memory usage of '{name}' from {format_memory_size(before)} to {format_memory_size(after)} "
          f"({100*(before-before)/before:.1f}% reduction)")

# ---------------------------- Imputation ----------------------------

def imputerColumn(data: DataFrame, column: str, strategy: str, **kwargs):
    if column not in data.columns:
        print(f"Column {column} not found")
        return data

    missing_before = data[column].isna().sum()
    print(f"Imputing {column} using {strategy} (missing={missing_before})")

    if strategy in ["median", "mean", "most_frequent"]:
        imputer = SimpleImputer(strategy=strategy)
        data[[column]] = imputer.fit_transform(data[[column]])

    elif strategy == "sentinel":
        fill_value = kwargs.get("fill_value", -999 if np.issubdtype(data[column].dtype, np.number) else "MISSING")
        data[column] = data[column].fillna(fill_value)

    elif strategy == "iterative":
        num_cols = [column] if np.issubdtype(data[column].dtype, np.number) else []
        if num_cols:
            iterative_imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=kwargs.get("n_estimators",50),
                    random_state=42
                ),
                max_iter=kwargs.get("max_iter",10),
                random_state=42
            )
            data[num_cols] = iterative_imputer.fit_transform(data[num_cols])
    else:
        print(f"Unknown strategy: {strategy}")

    missing_after = data[column].isna().sum()
    print(f"Missing after imputation: {missing_after}")
    return data

# ---------------------------- Outlier Removal ----------------------------

def remove_outliers(df: DataFrame, cols: list[str]):
    filtered_df = df.copy()
    for col in cols:
        if col not in filtered_df.columns or not pd.api.types.is_numeric_dtype(filtered_df[col]):
            continue

        Q1, Q3 = filtered_df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        if IQR == 0:
            continue

        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        mask_keep = (filtered_df[col] >= lower) & (filtered_df[col] <= upper)
        outliers_count = (~mask_keep).sum()

        print(f"{col}: removed {outliers_count} outliers ({outliers_count/len(filtered_df)*100:.2f}%)")
        filtered_df = filtered_df[mask_keep].copy()

        # Optional plot
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
        ax1.hist(df[col], bins=30, color='red', alpha=0.6)
        ax1.set_title(f'Before: {col}')
        ax2.hist(filtered_df[col], bins=30, color='green', alpha=0.6)
        ax2.set_title(f'After: {col}')
        plt.show()

    return filtered_df

# ---------------------------- Feature Engineering ----------------------------

def featuresEng(prior_df, orders_df, train_df):
    print("Creating features from PRIOR data")

    user_features = prior_df.groupby('user_id').agg(
        user_total_orders=('order_id','nunique'),
        user_total_items=('product_id','count'),
        user_reorder_rate=('reordered','mean')
    ).reset_index()

    product_features = prior_df.groupby('product_id').agg(
        product_orders=('order_id','nunique'),
        product_users=('user_id','nunique')
    ).reset_index()

    up_features = prior_df.groupby(['user_id','product_id']).agg(
        up_count=('order_id','count'),
        up_last_order=('order_number','max')
    ).reset_index()

    time_features = orders_df[['order_id','order_dow','order_hour_of_day']].copy()
    time_features.columns = ['order_id','order_day','order_hour']

    features = train_df.copy()
    features = features.merge(user_features, on='user_id', how='left')
    features = features.merge(product_features, on='product_id', how='left')
    features = features.merge(up_features, on=['user_id','product_id'], how='left')
    if 'order_id' in features.columns:
        features = features.merge(time_features, on='order_id', how='left')

    # Non-linear features
    features['orders_x_count'] = features['user_total_orders'] * features['up_count']
    features['log_up_count'] = np.log1p(features['up_count'])
    features['user_orders_x_product_orders'] = features['user_total_orders'] * features['product_orders']

    print(f"Created {features.shape[1]-4} features total, shape: {features.shape}")
    return features

# ---------------------------- Feature Helpers ----------------------------

def get_feature_names(df):
    exclude_cols = ['user_id','product_id','order_id','reordered']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"Feature columns ({len(feature_cols)} total): {feature_cols}")
    return feature_cols

def feature_scaling(data: DataFrame, columns_to_scale: list[str], method='standard', scaler=None, fit=True):
    if method=='standard':
        scaler = scaler or StandardScaler()
    elif method=='minmax':
        scaler = scaler or MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    if fit:
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    else:
        data[columns_to_scale] = scaler.transform(data[columns_to_scale])
    return data, scaler


def multicollinearity(df, feature_cols, corr_threshold=0.85, sample_size=5000, show_plot=False):
    """
    Remove highly correlated features to reduce multicollinearity
    """
    # Sample data for speed if dataset is large
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    # Calculate correlation matrix
    corr_matrix = df_sample[feature_cols].corr().abs()

    if show_plot:
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Correlation')
        plt.title(f'Feature Correlation Matrix (Threshold: {corr_threshold})')
        plt.xticks(range(len(feature_cols)), feature_cols, rotation=90)
        plt.yticks(range(len(feature_cols)), feature_cols)
        plt.tight_layout()
        plt.show()

    # Find features to remove
    columns_to_remove = set()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > corr_threshold:
                colname_i = corr_matrix.columns[i]
                colname_j = corr_matrix.columns[j]
                high_corr_pairs.append((colname_i, colname_j, corr_matrix.iloc[i, j]))

                # Keep the first one, remove the second
                if colname_j not in columns_to_remove:
                    columns_to_remove.add(colname_j)

    # Create cleaned feature list
    cleaned_features = [col for col in feature_cols if col not in columns_to_remove]

    # Report results
    if high_corr_pairs:
        print(f"\nRemoved {len(columns_to_remove)} features due to high correlation (> {corr_threshold}):")
        for col in sorted(columns_to_remove):
            print(f"  - {col}")

        print(f"\nHigh correlation pairs found:")
        for col1, col2, corr in high_corr_pairs[:10]:  # Show first 10
            print(f"  {col1} - {col2}: {corr:.3f}")
        if len(high_corr_pairs) > 10:
            print(f"  ... and {len(high_corr_pairs) - 10} more pairs")
    else:
        print(f"\nNo features removed. No correlations above {corr_threshold} threshold.")

    return cleaned_features, columns_to_remove, high_corr_pairs


def get_top_correlations(df, feature_cols, top_n=10):
    """
    Get top correlations between features
    """
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr().abs()

    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find top correlations
    top_corrs = []
    for col in upper.columns:
        for idx in upper.index:
            if not pd.isna(upper.loc[idx, col]):
                top_corrs.append((col, idx, upper.loc[idx, col]))

    # Sort by correlation value
    top_corrs.sort(key=lambda x: x[2], reverse=True)

    return top_corrs[:top_n]
