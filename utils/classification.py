from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

__all__ = (
    "linear_regressor",
    "lasso_classifier",
    "ridge_classifier",
    "knn_classifier",
    "svm_linear_classifier",
    "svm_kernel_classifier",
    "decision_tree_classifier",
    "random_forest_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
)


# noinspection PyPep8Naming
def linear_regressor(X_train, y_train):
    return LinearRegression().fit(X_train, y_train)


# lasso has no proper classifier, so we'll just bodge one like so
# noinspection PyPep8Naming
def lasso_classifier(X_train, y_train, alpha: float):
    C = 1 / alpha if alpha > 0 else 1.0
    return LogisticRegression(
        penalty="l1",
        C=C,
        solver="liblinear",
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def ridge_classifier(X_train, y_train, alpha: float):
    return RidgeClassifier(alpha=alpha).fit(X_train, y_train)


# noinspection PyPep8Naming
def knn_classifier(X_train, y_train, n_neighbors: int):
    return KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)


# noinspection PyPep8Naming
def svm_linear_classifier(X_train, y_train, C: float, max_iter):
    return LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
from sklearn.svm import LinearSVC
# or SVC with linear kernel

def svm_kernel_classifier(X_train, y_train, C: float): # Linear SVM faster the RBF
    return LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=10000,
        dual=False if X_train.shape[0] > X_train.shape[1] else True
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def decision_tree_classifier(X_train, y_train, max_depth=None):
    return DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def random_forest_classifier(X_train, y_train, n_estimators, max_depth=None):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def xgboost_classifier(
    X_train,
    y_train,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        device=device,
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def lightgbm_classifier(
    X_train,
    y_train,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str,
):
    return LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        device=device,
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)
