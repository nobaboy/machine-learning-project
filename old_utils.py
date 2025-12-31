from pandas import DataFrame

from utils.visualization import plot_correlation_heatmap


def get_feature_names(df):
    exclude_cols = ['user_id', 'product_id', 'order_id', 'reordered']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nFeature columns ({len(feature_cols)} total):")
    for col in feature_cols:
        print(f" - {col}")

    return feature_cols


def multicollinearity(
    df: DataFrame,
    feature_cols: list,
    corr_threshold: float = 0.85,
    sample_size: int = 10000,
    plot: bool = True,
):
    print(f"Checking multicollinearity for {len(feature_cols)} features ")

    # Calculate correlation matrix
    print("Calculating correlation matrix: ")
    corr_matrix = df[feature_cols].corr().abs()

    # Create visualization BEFORE removing features
    if plot and len(feature_cols) > 1:
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
    if plot and len(features_to_keep) > 1 and len(features_to_keep) <= 20:
        remaining_corr = df[features_to_keep].corr().abs()
        plot_correlation_heatmap(
            remaining_corr,
            title="Feature Correlation Heatmap After Remove",
            figsize=(10, 8),
        )

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
