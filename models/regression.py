from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

__all__ = (
    "lasso_regressor",
    "ridge_regressor",
    "elastic_net_regressor",
    "svm_linear_regressor",
    "svm_kernel_regressor",
    "knn_regressor",
    "decision_tree_regressor",
    "random_forest_regressor",
    "xgboost_regressor",
)


# noinspection PyPep8Naming
def lasso_regressor(X_train, y_train, alpha: float):
    return Lasso(alpha=alpha, random_state=42).fit(X_train, y_train)


# noinspection PyPep8Naming
def ridge_regressor(X_train, y_train, alpha: float):
    return Ridge(alpha=alpha, random_state=42).fit(X_train, y_train)


# noinspection PyPep8Naming
def elastic_net_regressor(X_train, y_train, alpha: float, l1_ratio: float):
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def svm_linear_regressor(
    X_train,
    y_train,
    C: float,
    epsilon: float = 0.05,
    max_samples: int = None,
):
    if max_samples and len(X_train) > max_samples:
        X_train = X_train.sample(max_samples, random_state=42)
        y_train = y_train.loc[X_train.index]

    return LinearSVR(
        C=C,
        epsilon=epsilon,
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def svm_kernel_regressor(
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
        cache_size=1000,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def knn_regressor(
    X_train,
    y_train,
    n_neighbors: int = 5,
    weights: str = "distance",
):
    return KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def decision_tree_regressor(
    X_train,
    y_train,
    max_depth: int | None = None,
):
    return DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
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


# noinspection PyPep8Naming
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
        tree_method="hist",
        device=device,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        random_state=42,
    ).fit(X_train, y_train)


# TODO lightgbm
