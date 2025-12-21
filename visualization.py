import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

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
