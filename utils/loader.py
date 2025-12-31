import numpy as np
import pandas as pd
from pandas import DataFrame

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

            # we only want to convert low cardinality columns (e.g. eval_set from orders dataset)
            if unique / total < 0.5:
                df[col] = df[col].astype("category")

    after = calculate_memory_usage(df)
    pct = 100 * (before - after) / before
    print(f"Reduced memory usage of '{name}' from {format_memory_size(before)} to {format_memory_size(after)} ({pct:.1f}% reduction)")
