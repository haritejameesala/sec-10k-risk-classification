import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from catboost import CatBoostClassifier

from .utils import get_logger, MODELS_DIR, DATA_DIR, LABELS, Timer, save_json
from .features import build_features, FeatureBuilder

logger = get_logger("train")

XGBOOST_PARAMS = dict(
    n_estimators=300,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=15,
    gamma=0.3,
    reg_alpha=0.5,
    reg_lambda=5,
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)

ADABOOST_PARAMS = dict(
    estimator=DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=10
    ),
    n_estimators=50,
    learning_rate=0.3,
    random_state=42,
)

CATBOOST_PARAMS = dict(
    iterations=600,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=10,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=0,
    thread_count=-1,
)

MODELS_CONFIG = {
    "xgboost": {
        "class": xgb.XGBClassifier,
        "params": XGBOOST_PARAMS,
    },
    "adaboost": {
        "class": AdaBoostClassifier,
        "params": ADABOOST_PARAMS,
    },
    "catboost": {
        "class": CatBoostClassifier,
        "params": CATBOOST_PARAMS,
    },
}

_FEATURE_COLS = ["text", "clean_risk", "clean_mda", "clean_business", "clean_financials"]

def load_processed_data(csv_path: Path = None) -> tuple[pd.DataFrame, list]:
 
    csv_path = csv_path or (DATA_DIR / "processed.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run preprocess.py first."
        )

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])

    for col in _FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""

    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    le = LabelEncoder()
    labels = le.fit_transform(df["label"]).tolist()
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    logger.info(f"Label classes: {le.classes_}")

    return df[_FEATURE_COLS].reset_index(drop=True), labels

def train_all_models(
    df: pd.DataFrame,
    labels: list[int],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
   
    logger.info(f"Starting training pipeline. Samples: {len(df)}")

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)
    y_train  = np.array([labels[i] for i in train_idx])
    y_test   = np.array([labels[i] for i in test_idx])

    logger.info(f"Split -> train: {len(y_train)}, test: {len(y_test)}")

    logger.info("Building features ...")
    with Timer() as t:
        X_train, X_test, fb = build_features(df_train, df_test)
    logger.info(f"Features built in {t}.")

    trained_models = {}

    for name, cfg in MODELS_CONFIG.items():
        logger.info(f"Training {name.upper()} ...")
        clf = cfg["class"](**cfg["params"])

        if name == "adaboost":
            X_tr = X_train.toarray()
            X_te = X_test.toarray()
        elif name == "catboost":
            X_tr = X_train.astype(np.float32).toarray()
            X_te = X_test.astype(np.float32).toarray()
        else:
            X_tr = X_train
            X_te = X_test

        with Timer() as t:
            if name == "xgboost":
                clf.fit(
                    X_tr, y_train,
                    eval_set=[(X_te, y_test)],
                    verbose=False,
                )
            elif name == "catboost":
                clf.fit(
                    X_tr, y_train,
                    eval_set=(X_te, y_test),
                )
            else:
                clf.fit(
                    X_tr, y_train,
                )

        preds = clf.predict(X_tr)
        if name == "catboost":
            preds = preds.flatten()
        train_acc = float((preds == y_train).mean())

        if name == "xgboost":
	        imp = clf.feature_importances_

	        print("\nXGBoost Feature Importance Distribution")
	        print(
		        np.percentile(
			        imp,
			        [50, 75, 90, 95, 99, 99.9]
		        )
	        )

	        print(
		        f"Non-zero importance features: "
		        f"{(imp > 0).sum()} / {len(imp)}"
	        )
		
        logger.info(
            f"{name.upper()} trained in {t}. "
            f"Train accuracy: {train_acc:.4f}"
        )

        model_path = MODELS_DIR / f"{name}_model.joblib"
        joblib.dump(clf, model_path)
        logger.info(f"Saved -> {model_path}")

        trained_models[name] = {
            "model": clf,
            "X_test": X_te,
            "train_acc": train_acc,
        }

    split_info = {
        "test_size": test_size,
        "n_train": int(len(y_train)),
        "n_test":  int(len(y_test)),
        "random_seed": random_seed,
    }
    save_json(split_info, MODELS_DIR / "split_info.json")

    np.save(MODELS_DIR / "y_test.npy", y_test)
    sp.save_npz(str(MODELS_DIR / "X_test_xgb.npz"), X_test)
    np.save(MODELS_DIR / "X_test_ada.npy", X_test.toarray())
    np.save(MODELS_DIR / "X_test_cat.npy", X_test.astype(np.float32).toarray())

    return {
        "models": trained_models,
        "splits": (df_train, df_test, y_train, y_test),
        "feature_builder": fb,
    }

if __name__ == "__main__":
    df, labels = load_processed_data()
    results = train_all_models(df, labels)
    logger.info("All models trained and saved successfully.")