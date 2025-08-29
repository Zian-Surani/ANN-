
import os, json, joblib
import numpy as np, pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "ids_ann_model.joblib")
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "ids_preproc_pipe.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "ids_metrics.json")

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    ann = MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", solver="adam",
                        max_iter=30, random_state=42)
    pipe = Pipeline(steps=[("preproc", preproc), ("clf", ann)])
    return pipe

def train_and_save(X: pd.DataFrame, y_bin: pd.Series) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    try:
        proba = pipe.predict_proba(X_test)[:,1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = float("nan")

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    metrics = {"roc_auc": auc, "report": report, "confusion_matrix": cm, "samples": int(len(X))}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train first.")
    return joblib.load(MODEL_PATH)

def predict_one(pipe: Pipeline, row: dict) -> Dict[str, Any]:
    import pandas as pd
    X = pd.DataFrame([row])
    pred = int(pipe.predict(X)[0])
    out = {"binary": pred, "label": "attack" if pred==1 else "normal"}
    try:
        proba = float(pipe.predict_proba(X)[:,1][0])
        out["attack_prob"] = proba
    except Exception:
        out["attack_prob"] = None
    return out

def predict_batch(pipe: Pipeline, df: pd.DataFrame):
    preds = pipe.predict(df)
    try:
        probs = pipe.predict_proba(df)[:,1]
    except Exception:
        probs = None
    out = df.copy()
    out["binary_pred"] = preds
    out["label_pred"] = np.where(preds==1,"attack","normal")
    if probs is not None:
        out["attack_prob"] = probs
    return out
