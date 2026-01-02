import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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
        # if all fails, treat prediction as the score (e.g. LinearRegression)
        y_score = y_pred

    if len(np.unique(y_test)) < 2:
        auc = float("nan")
        precision = float("nan")
    else:
        auc = float(roc_auc_score(y_test, y_score))
        precision = float(average_precision_score(y_test, y_score))

    metrics = {
        #       run the code and get the error to further understand it since I didn't copy it
        # FIXME after running the models, trying to calculate the accuracy for some models that don't
        #       have `predict_proba` will fail since it's trying to compare continuous and binary values
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": auc,
        "avg_precision": precision,
    }

    print(f"\nMetrics for {name}:")
    for k, v in metrics.items():
        print(f"{k:<16}: {v:.3f}")

    if plot:
        ...

    return metrics
