import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def visualize_memory_usage(before: int, after: int):
    labels = ["Before", "After"]
    sizes = [before / (1024 ** 2), after / (1024 ** 2)] # convert each size to mib since we have it in bytes

    plt.figure(figsize=(8, 5))
    axes = sns.barplot(x=labels, y=sizes, hue=labels, palette="viridis")

    # TODO is there a better way of drawing the size ontop of its bar
    for container in axes.containers:
        axes.bar_label(container, fmt="%.1f MiB")

    plt.title("Dataset Memory Optimization")
    plt.ylabel("Memory Usage (MiB)")

    plt.show()


def analyze_and_visualize_missing(df: DataFrame) -> list[str] | None:
    missing = df.isnull().sum()

    if missing.empty:
        print("\nThe dataset does not have any missing values")
        return None

    print("\nNumber of missing rows for each feature:")
    for col in missing.index:
        pct = 100 * (missing[col] / len(df))
        print(f"{col:<24} {missing[col]:>8} rows ({pct:.1f}% of the total dataset)")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.index.tolist(), y=missing.values)
    plt.title("Missing Data") # TODO better title
    plt.ylabel("Missing Rows (millions)") # TODO better title

    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)

    plt.show()

    return missing[missing > 0].index.tolist()
def plot_correlation_heatmap(corr_matrix, title="Correlation Heatmap", figsize=(12, 10)): # AI
    plt.figure(figsize=figsize)

    # Create mask for upper triangle (optional)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                cbar_kws={"shrink": 0.8},
                linewidths=0.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_top_correlations(high_corr_pairs, top_n=10): # AI
    if not high_corr_pairs:
        print("No highly correlated pairs found")
        return

    # Sort by correlation
    sorted_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:top_n]

    # Prepare data
    labels = [f"{p[0]}\n↔\n{p[1]}" for p in sorted_pairs]
    correlations = [p[2] for p in sorted_pairs]

    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sorted_pairs)), correlations,
                   color=['red' if corr > 0.9 else 'orange' if corr > 0.8 else 'yellow'
                          for corr in correlations])

    plt.title(f'Top {len(sorted_pairs)} Most Correlated Feature Pairs',
              fontsize=14, fontweight='bold')
    plt.xlabel('Feature Pairs', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xticks(range(len(sorted_pairs)), labels, rotation=45, ha='right')
    plt.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Threshold (0.85)')

    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{corr:.3f}', ha='center', va='bottom', fontsize=9)

    plt.legend()
    plt.tight_layout()
    plt.show()
