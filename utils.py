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

            # we only want to convert high cardinality columns (e.g. eval_set from orders dataset)
            if unique / total < 0.5:
                df[col] = df[col].astype("category")

    after = calculate_memory_usage(df)
    diff = 100 * (before - after) / before
    print(f"Reduced memory usage of '{name}' from {format_memory_size(before)} to {format_memory_size(after)} ({diff:.1f}% reduction)")
def imputerColumn(data: DataFrame, column: str, strategy: str, **kwargs):
    if column not in data.columns: # Make sure column in the DataFram
        print(f"Column {column} not found")
        return data

    print(f"Imputing column {column} with strategy {strategy}")
    print(f"Missing values before: {data[column].isna().sum()}")

    #--------Simple strategies---------
    if strategy in ["median", "mean","most_frequent"]:
        imputer = SimpleImputer(strategy=strategy)
        data[[column]] = imputer.fit_transform(data[[column]]) # [[]] cause impute only work with two dimensional list || we can u .tolist

    #-------- Sentinel-------
    elif strategy == "sentinel":
        # choose sentinel value based on dtype
        if np.issubdtype(data[column].dtype, np.integer):
            sentinel = kwargs.get("fill_value", -999)
        elif np.issubdtype(data[column].dtype, np.floating):
            sentinel = kwargs.get("fill_value", -999.0)
        else:
            sentinel = kwargs.get("fill_value", "MISSING")
        data[column] = data[column].fillna(sentinel)

    # sentinel do not work with (linearRegression , KNN) because the fictitious value distorts the calculations


    # The iterative is a way to fill Nan values but with predect the Nan values using model like ( Regression , Random forset or Iterative imputer )
    #--------Iterative Model-Based--------
    elif strategy == "iterative" :
        numCol = data.select_dtypes(include=[np.number]).columns
        iterative_imputer = IterativeImputer(
            estimator= RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 50),
                random_state=42
        ),
            maxIter=kwargs.get("max_iter", 10),
            random_state=42
        )
    # Apply iterative work to numeric columns

    data[numCol] = iterative_imputer.fit_transform(data[numCol])

    # Make sure we onlay apply on numeric columns

    data[numCol] = iterative_imputer.fit_transform(data[numCol])
    print(f"Missing values after: {data[numCol].isna().sum()}")
    return data
