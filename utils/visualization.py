import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
import numpy as np

__all__ = (
    "visualize_memory_usage",
    "analyze_and_visualize_missing",
    "visualize_outlier_removal",
    "visualize_numerical_correlation",
    "visualize_top_correlations",
)


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def visualize_memory_usage(before: int, after: int):
    labels = ["Before", "After"]

    # convert each size to mib since we have it in bytes
    sizes = np.array([before, after]) / 1024 ** 2

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
    plt.title("Missing Data")
    plt.ylabel("Missing Rows")

    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)
    plt.show()

    return missing[missing > 0].index.tolist()


# TODO fix this to show either winsorization or removal
def visualize_outlier_removal(
    df: DataFrame,
    col: str,
    mask_keep: Series,
    outliers_count: int,
    before_count: int
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[col], bins=30, color="red", alpha=0.6, ax=ax1)
    ax1.set_title("Before")
    ax1.set_xlabel(col)

    # After
    sns.histplot(df[mask_keep][col], bins=30, color="green", alpha=0.6, ax=ax2)
    ax2.set_title("After")
    ax2.set_xlabel(col)

    pct = 100 * outliers_count / before_count
    plt.suptitle(f"Outlier Removal: {col}\nRemoved {outliers_count:,} outliers ({pct:.1f}%)")
    plt.tight_layout()
    plt.show()



def visualize_numerical_correlation(corr_matrix, title: str, figsize=(12, 10)):
    plt.figure(figsize=figsize)

    # create mask to hide upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def visualize_top_correlations(
    high_corr_pairs,
    top_n: int = 10,
    threshold: float = 0.85,
    figsize=(12, 6),
):
    if not high_corr_pairs:
        print("No highly correlated feature pairs found")
        return

    plt.figure(figsize=figsize)

    # sort by correlation
    sorted_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:top_n]

    labels = [f"{f1}\n↔\n{f2}" for f1, f2, _ in sorted_pairs]
    correlations = [corr for _, _, corr in sorted_pairs]

    colors = ["red" if c > 0.9 else "orange" if c > 0.8 else "yellow" for c in correlations]
    bars = sns.barplot(x=labels, y=correlations, hue=labels, palette=colors)

    plt.axhline(
        y=threshold,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Threshold {threshold}",
    )

    plt.title(
        f"Top {len(correlations)} Most Correlated Feature Pairs",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Feature Pairs")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")

    # TODO surely there is a better way of doing this rather than accessing patches, could just use `plt.subplots`
    for bar, corr in zip(bars.patches, correlations):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{corr:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.legend()
    plt.tight_layout()
    plt.show()
