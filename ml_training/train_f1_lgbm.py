# ml_training/train_f1_lgbm.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "output", "training_f1_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "ml", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "f1_lgbm.txt")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna()


def prepare_data(df: pd.DataFrame):
    y = df["assigned_optimal"].astype(int)

    feature_cols = [
        "driver",
        "period",
        "current_load_period",
        "need_period",
        "need_gap",
        "total_workers",
    ]

    X = df[feature_cols].astype(float)
    return X, y, feature_cols


def train_lgbm_classifier(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    y_pred_proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)

    print(f"[f1] AUC: {auc:.4f} | Accuracy: {acc:.4f}")

    return booster


if __name__ == "__main__":
    print(f"[f1] Lendo dataset em {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    X, y, feature_cols = prepare_data(df)

    print(f"[f1] Amostras: {len(df)}")
    print(f"[f1] Features: {feature_cols}")

    booster = train_lgbm_classifier(X, y)
    booster.save_model(MODEL_PATH)
    print(f"[f1] Modelo salvo em: {MODEL_PATH}")
