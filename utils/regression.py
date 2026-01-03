from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

__all__ = (
    "ols_regressor",
    "lasso_regressor",
    "ridge_regressor",
    "elasticnet_regressor",
    "svr_linear_regressor",
    "svr_kernel_regressor",
    "knn_regressor",
    "decision_tree_regressor",
    "random_forest_regressor",
    "xgboost_regressor",
)


# Ordinary Least Squares
def ols_regressor(X_train, y_train):
    return LinearRegression().fit(X_train, y_train)


# L1 Regularization
def lasso_regressor(X_train, y_train, alpha: float):
    return Lasso(alpha=alpha, random_state=42).fit(X_train, y_train)


# L2 Regularization
def ridge_regressor(X_train, y_train, alpha: float):
    return Ridge(alpha=alpha, random_state=42).fit(X_train, y_train)


# L1 + L2
def elasticnet_regressor(X_train, y_train, alpha: float, l1_ratio: float):
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
    ).fit(X_train, y_train)


# Support Vector Regressor — Linear
def svr_linear_regressor(
    X_train,
    y_train,
    C: float,
    epsilon: float = 0.05,
    max_samples: int = 15000,
):
    # Subsample for scalability
    if len(X_train) > max_samples:
        X_train = X_train.sample(max_samples, random_state=42)
        y_train = y_train.loc[X_train.index]

    return SVR(
        kernel="linear",
        C=C,
        epsilon=epsilon,
        cache_size=1000,  # MB, speeds up kernel computation
    ).fit(X_train, y_train)

# Support Vector Regressor — RBF Kernel
from sklearn.svm import SVR

def svr_kernel_regressor(
    X_train,
    y_train,
    C: float,
    gamma="scale",
    epsilon: float = 0.1,
    max_samples: int = 8000,
):
    # Subsample for scalability
    if len(X_train) > max_samples:
        X_train = X_train.sample(max_samples, random_state=42)
        y_train = y_train.loc[X_train.index]

    return SVR(
        kernel="rbf",
        C=C,
        gamma=gamma,
        epsilon=epsilon,
        cache_size=1000,  # MB
    ).fit(X_train, y_train)



# K-Nearest Neighbors Regressor
def knn_regressor(X_train, y_train, n_neighbors: int = 5, weights: str = "distance"):
    return KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1,
    ).fit(X_train, y_train)


# Decision Tree Regressor
def decision_tree_regressor(X_train, y_train, max_depth: int | None = None):
    return DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42,
    ).fit(X_train, y_train)


# Random Forest Regressor
def random_forest_regressor(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int | None = None,
):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)


# Gradient Boosting Regressor — XGBoost

def xgboost_regressor(
    X_train,
    y_train,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    device: str = "cuda",
):
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        random_state=42,

        # 🔥 GPU SETTINGS
        tree_method="hist",
        device="cuda",

        # Performance
        subsample=0.8,
        colsample_bytree=0.8,

        # CPU fallback control
        n_jobs=1,
    ).fit(X_train, y_train)
