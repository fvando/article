# ml_training/train_f2_lgbm.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "output", "training_f2_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "ml", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "f2_lgbm.txt")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.dropna()


def prepare_data(df: pd.DataFrame):
    y = df["improvement"].astype(float)

    feature_cols = [
        "neighborhood_size",
        "removed_assignments",
        "objective_before",
        "relaxation_level",
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].astype(float)
    return X, y, feature_cols


def train_lgbm_regressor(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": ["l2", "l1"],
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

    y_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)

    print(f"[f2] RMSE: {rmse:.4f} | R²: {r2:.4f}")

    return booster


if __name__ == "__main__":
    print(f"[f2] Lendo dataset em {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    X, y, feature_cols = prepare_data(df)

    print(f"[f2] Amostras: {len(df)}")
    print(f"[f2] Features: {feature_cols}")

    booster = train_lgbm_regressor(X, y)
    booster.save_model(MODEL_PATH)
    print(f"[f2] Modelo salvo em: {MODEL_PATH}")
