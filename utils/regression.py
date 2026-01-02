from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR

__all__ = (
    "create_ols_regressor",
    "create_lasso_regressor",
    "create_ridge_regressor",
    "create_elasticnet_regressor",
    "create_svr_linear_regressor",
    "create_svr_kernel_regressor",
)


# Ordinary Least Squares
def create_ols_regressor(X_train, y_train):
    return LinearRegression().fit(X_train, y_train)


# L1 Regularization
def create_lasso_regressor(X_train, y_train, alpha: float):
    return Lasso(alpha=alpha, random_state=42).fit(X_train, y_train)


# L2 Regularization
def create_ridge_regressor(X_train, y_train, alpha: float):
    return Ridge(alpha=alpha, random_state=42).fit(X_train, y_train)


# L1 + L2
def create_elasticnet_regressor(X_train, y_train, alpha: float, l1_ratio: float):
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
    ).fit(X_train, y_train)


# Support Vector Regressor — Linear
def create_svr_linear_regressor(X_train, y_train, C: float):
    return SVR(
        kernel="linear",
        C=C,
    ).fit(X_train, y_train)


# Support Vector Regressor — RBF Kernel
def create_svr_kernel_regressor(X_train, y_train, C: float, gamma="scale"):
    return SVR(
        kernel="rbf",
        C=C,
        gamma=gamma,
    ).fit(X_train, y_train)
