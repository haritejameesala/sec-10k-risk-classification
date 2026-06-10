import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .utils import MODELS_DIR, get_logger, save_json

logger = get_logger("evaluate")


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)

    if hasattr(preds, "flatten"):
        preds = preds.flatten()

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(
            precision_score(y_test, preds, average="weighted")
        ),
        "recall": float(
            recall_score(y_test, preds, average="weighted")
        ),
        "f1": float(
            f1_score(y_test, preds, average="weighted")
        ),
    }

    cm = confusion_matrix(y_test, preds)

    report = classification_report(
        y_test,
        preds,
        output_dict=True
    )

    return metrics, cm.tolist(), report


def load_test_data(model_name):
    y_test = np.load(MODELS_DIR / "y_test.npy")

    if model_name == "xgboost":
        X_test = sp.load_npz(MODELS_DIR / "X_test_xgb.npz")

    elif model_name == "adaboost":
        X_test = np.load(MODELS_DIR / "X_test_ada.npy")

    elif model_name == "catboost":
        X_test = np.load(MODELS_DIR / "X_test_cat.npy")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return X_test, y_test


def main():

    results = {}

    best_model = None
    best_f1 = -1

    for model_name in ["xgboost", "adaboost", "catboost"]:

        logger.info(f"Evaluating {model_name}")

        model = joblib.load(
            MODELS_DIR / f"{model_name}_model.joblib"
        )

        X_test, y_test = load_test_data(model_name)

        metrics, cm, report = evaluate_model(
            model_name,
            model,
            X_test,
            y_test
        )

        results[model_name] = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        logger.info(
            f"{model_name} | "
            f"Acc={metrics['accuracy']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model_name

    save_json(
        results,
        MODELS_DIR / "evaluation_results.json"
    )

    save_json(
        {
            "best_model": best_model,
            "best_f1": best_f1,
        },
        MODELS_DIR / "best_model.json"
    )

    print("\n")
    print("=" * 60)
    print(f"BEST MODEL : {best_model.upper()}")
    print(f"BEST F1    : {best_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()