from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

__all__ = (
    "logistic_classifier",
    "lasso_classifier",
    "ridge_classifier",
    "svm_linear_classifier",
    "svm_kernel_classifier",
    "knn_classifier",
    "decision_tree_classifier",
    "random_forest_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
)


# noinspection PyPep8Naming
def logistic_classifier(X_train, y_train, C: float, max_iter: int):
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
    ).fit(X_train, y_train)


# lasso has no proper classifier, so we'll just bodge one like so
# noinspection PyPep8Naming
def lasso_classifier(X_train, y_train, alpha: float):
    C = 1 / alpha if alpha > 0 else 1.0
    return LogisticRegression(
        C=C,
        penalty="l1",
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def ridge_classifier(X_train, y_train, alpha: float):
    return RidgeClassifier(alpha=alpha, random_state=42).fit(X_train, y_train)


# noinspection PyPep8Naming
def svm_linear_classifier(
    X_train,
    y_train,
    C: float,
    max_iter: int = 1000,
):
    return LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
# even though we have the option to not subsample, ideally we should since this
# model takes forever to train with large datasets like ours
def svm_kernel_classifier(
    X_train,
    y_train,
    C: float,
    kernel: str,
    max_iter: int = -1,
    max_samples: int | None = 10000,
):
    if max_samples and len(X_train) > max_samples:
        X_train = X_train.sample(max_samples, random_state=42)
        y_train = y_train.loc[X_train.index]

    return SVC(
        C=C,
        kernel=kernel,
        class_weight="balanced",
        max_iter=max_iter,
        cache_size=1024,
        random_state=42,
        probability=True,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def knn_classifier(X_train, y_train, n_neighbors: int):
    return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1).fit(X_train, y_train)


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
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def xgboost_classifier(
    X_train,
    y_train,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    device: str = "cuda",
):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        device=device,
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
    ).fit(X_train, y_train)


# noinspection PyPep8Naming
def lightgbm_classifier(
    X_train,
    y_train,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    num_leaves: int = 30,
    device: str = "gpu",
):
    return LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        device=device,
        n_jobs=-1,
        verbose=-1,
        random_state=42,
    ).fit(X_train, y_train)
