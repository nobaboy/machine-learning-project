# import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.imputer import SimpleImputer
# from pandas import DataFrame


def load_data(file_name: str):
    return pd.read_csv(file_name)

def obInToCat(list):
    for col in list:
        Proir[col] = Proir[col].astype('category')

def imputer_median(Data,columnName):
    if columnName in Data.columns:
        imputer = SimpleImputer(strategy='median')

        Data[[columnName]] = imputer.fit_transform(Data[[columnName]])

if __name__ == "__main__":

    #load datasets 
    aisles = load_data("data/aisles.csv")
    departments = load_data("data/departments.csv")
    prior = load_data("data/order_products__prior.csv")
    train = load_data("data/order_products__train.csv")
    orders = load_data("data/orders.csv")
    products = load_data("data/products.csv")

    #--------visualization--------
    Datasets = [aisles,departments,prior,train,orders,products]

    for data in Datasets : 
        data.hist(bins=20,figsize=(12,4))

    #--------Join The Datasets--------

    #join prior data 
    Proir = (
        proir 
        .merge(orders, on="order_id", how = "inner")
        .merge(products, on="products_id", how = "inner")
        .merge(departments, on="departments_id", how = "inner")
        .merge(aisles, on="aisle_id", how = "inner")
    )

    #join train data 
    Train = (
        train 
        .merge(orders, on="order_id", how = "inner")
        .merge(products, on="products_id", how = "inner")
        .merge(departments, on="departments_id", how = "inner")
        .merge(aisles, on="aisle_id", how = "inner")
    )
    
    #--------Memory Optimization--------

    # step one 
    Proir['order_id'] = Proir['order_id'].astype(int32)
    Proir['product_id'] = Proir['product_id'].astype(int32)
    Proir['user_id'] = Proir['user_id'].astype(int32)

    Proir['aisle_id'] = Proir['ailse_id'].astype(int16)
    
    Proir['reordered_id'] = Proir['reordered_id'].astype(int8)
    Proir['departments_id'] = Proir['departments_id'].astype(int8)
    Proir['order_hour_of_day'] = Proir['order_hour_of_day'].astype(int8)

    #step two 
    obInToCat(['porducts','aisle','departments','eval_set'])


    #--------visualization--------
    Proir.hist(bins=20,figsize=(12,4))


    #--------Clean The data--------
    imputer_median[Proir,'days_since_prior_order']
