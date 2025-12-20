import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix


def load_data(Path):
    data = pd.read_csv(Path, engine="pyarrow")  # 30 - 50 %
    return data

def obj_to_cat(data, columns):
    for col in columns:
        if col in data.columns:
            data[col] = data[col].astype('category')
    return data  # Added return statement

def imputer_median(data, columnName):
    if columnName in data.columns:
        imputer = SimpleImputer(strategy='median')
        data[[columnName]] = imputer.fit_transform(data[[columnName]])
    return data  # Added return statement

def OutliersRemover(Data, col):
    if isinstance(col, str):
        col = [col]  # Convert single column name to list

    for cl in col:
        if cl in Data.columns and Data[cl].dtype in ['int64', 'float64', 'int32', 'float32']:
            Q1 = Data[cl].quantile(0.25)
            Q3 = Data[cl].quantile(0.75)
            IQT = Q3 - Q1

            #Avoid division by zero
            if IQT > 0:
                lower = Q1 - 1.5 * IQT
                upper = Q3 + 1.5 * IQT

                Data = Data[(Data[cl] >= lower) & (Data[cl] <= upper)]

    return Data


if __name__ == '__main__':
    # Load all datasets
    prior = load_data(r"C:\New folder\order_products__prior.csv")
    train = load_data(r"C:\New folder\order_products__train.csv")
    orders = load_data(r"C:\New folder\orders.csv")
    departments = load_data(r"C:\New folder\departments.csv")
    aisles = load_data(r"C:\New folder\aisles.csv")
    products = load_data(r"C:\New folder\products.csv")

    # 1. Merge PRIOR data
    prior_full = (
        prior
        .merge(orders, on='order_id', how='inner')
        .merge(products, on='product_id', how='inner')
        .merge(aisles, on='aisle_id', how='inner')
        .merge(departments, on='department_id', how='inner')
    )

    #Preparing labels from TRAIN (NO DATA LEAKAGE)

    # 2. Get labels from TRAIN
    train_labels = train[['order_id', 'product_id', 'reordered']]

    train_with_user = train_labels.merge(
        orders[['order_id', 'user_id']],  # Only take these two columns
        on='order_id',
        how='inner'
    )

    # 3. Create base user-product
    base_features = prior_full[['user_id', 'product_id']].drop_duplicates()

    # 4. FINAL MERGE Combine features with labels
    train_data = base_features.merge(
        train_with_user[['user_id', 'product_id', 'reordered']],
        on=['user_id', 'product_id'],
        how='left'  # Keep all pairs from prior
    )

    # Fill NaN with 0 (products not in last order)
    train_data['reordered'] = train_data['reordered'].fillna(0)

    # Memory optimization
    prior_full["order_id"] = prior_full["order_id"].astype('int32')
    prior_full["product_id"] = prior_full["product_id"].astype('int32')
    prior_full["user_id"] = prior_full["user_id"].astype('int32')

    prior_full["aisle_id"] = prior_full["aisle_id"].astype('int16')

    prior_full["reordered"] = prior_full["reordered"].astype('int8')
    prior_full["department_id"] = prior_full["department_id"].astype('int8')
    prior_full["order_hour_of_day"] = prior_full["order_hour_of_day"].astype('int8')

    train_data["user_id"] = train_data["user_id"].astype('int32')
    train_data["product_id"] = train_data["product_id"].astype('int32')
    train_data["reordered"] = train_data["reordered"].astype('int8')

    # change object columns to category
    prior_full = obj_to_cat(prior_full, ["product_name", "aisle", "department", "eval_set"])


    # Handle missing values in prior_full
    print(prior_full.isnull().sum())
    # Check for missing values
    missing_info = prior_full.isna().sum()
    missing_cols = missing_info[missing_info > 0]

    if not missing_cols.empty:
        for col, count in missing_cols.items():
            percent = (count / len(prior_full)) * 100

        if 'days_since_prior_order' in missing_cols.index:
            median_val = prior_full['days_since_prior_order'].median()
            prior_full['days_since_prior_order'] = prior_full['days_since_prior_order'].fillna(median_val)
    print(prior_full.isnull().sum())
    # Remove outliers
    columns_to_check = ['add_to_cart_order', 'days_since_prior_order']


    prior_full = OutliersRemover(prior_full, columns_to_check)

    print(train_data.info())
    print("=" * 50)
    print(prior_full.info())


