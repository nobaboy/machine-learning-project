import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer , SimpleImputer
from pandas import DataFrame

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
