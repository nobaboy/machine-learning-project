import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,

    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

__all__ = (
    "evaluate_classifier",
    "evaluate_regressor",
    "evaluate_regressor_accuracy"
)


# noinspection PyPep8Naming
def evaluate_classifier(
    model,
    X_test,
    y_test,
    name: str,
    plot: bool = True,
) -> dict[str, float]:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        # if all fails, treat prediction as the score
        y_score = y_pred

    if len(np.unique(y_test)) < 2:
        auc = float("nan")
        precision = float("nan")
    else:
        auc = float(roc_auc_score(y_test, y_score))
        precision = float(average_precision_score(y_test, y_score))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": auc,
        "avg_precision": precision,
    }

    print(f"\nClassification metrics for {name}:")
    for k, v in metrics.items():
        print(f"{k:<16} {v:.3f}")

    if plot:
        ...

    return metrics


# noinspection PyPep8Naming
def evaluate_regressor(
    model,
    X_test,
    y_test,
    name: str,
) -> dict[str, float]:
    y_pred = model.predict(X_test)

    metrics = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    print(f"\nRegression metrics for {name}:")
    for k, v in metrics.items():
        print(f"{k:<8} {v:.3f}")

    return metrics


# noinspection PyPep8Naming
def evaluate_regressor_accuracy(model, X, y, days: int = 3):
    y_pred = model.predict(X)
    return float(np.mean(np.abs(y_pred - y) <= days))
