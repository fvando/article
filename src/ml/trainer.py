# import os
# import pandas as pd
# import numpy as np
# from typing import Tuple, Dict, Any

# from lightgbm import LGBMClassifier


# # ============================================================
# #  PATHS
# # ============================================================

# BASE_DIR = os.path.dirname(__file__)
# DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# os.makedirs(MODELS_DIR, exist_ok=True)

# F1_DATASET_PATH = os.path.join(DATASETS_DIR, "f1_assignment_dataset.csv")
# F2_DATASET_PATH = os.path.join(DATASETS_DIR, "f2_neighborhood_dataset.csv")

# F1_MODEL_PATH = os.path.join(MODELS_DIR, "f1_lgbm.txt")
# F2_MODEL_PATH = os.path.join(MODELS_DIR, "f2_lgbm.txt")

# # ============================================================
# # Utils
# # ============================================================

# def safe_read_csv(path):
#     """
#     Lê um CSV com tratamento de erros.
#     Se não existir, retorna None ao invés de quebrar a aplicação.
#     """
#     if not os.path.exists(path):
#         return None
#     try:
#         return pd.read_csv(path)
#     except Exception as e:
#         print(f"[ML] ERRO ao ler dataset: {e}")
#         return None


# # ============================================================
# #  TRAINER F1 — Assignment Model
# # ============================================================

# def train_f1() -> Dict[str, Any]:
#     """
#     Treina o modelo LightGBM f₁ (assignment).
#     Retorna métricas e paths dos modelos.
#     """
#     # if not os.path.exists(F1_DATASET_PATH):
#     #     raise FileNotFoundError(f"Dataset f₁ não encontrado: {F1_DATASET_PATH}")

#     df = safe_read_csv(F1_DATASET_PATH)
    
#     if df is None:
#         msg = f"[ML] Dataset f₁ não encontrado em: {F1_DATASET_PATH}\n" \
#                 f"→ Gere o dataset antes de treinar (menu ML → Gerar Dataset)."
#         print(msg)
#         return {
#             "success": False,
#             "message": msg,
#             "model_path": None,
#         }


#     # Features usadas
#     feature_cols = [
#         "driver",
#         "period",
#         "need",
#         "local_load",
#         "driver_load",
#         "demand_gap"
#     ]

#     print(f"[ML] Treinando modelo f₁... ({len(df)} amostras)")

#     X = df[feature_cols].astype(float)
#     y = df["label"].astype(int)

#     # Modelo
#     clf = LGBMClassifier(
#         n_estimators=300,
#         learning_rate=0.05,
#         max_depth=-1,
#         num_leaves=64,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         objective="binary"
#     )

#     clf.fit(X, y)

#     # Métricas simples
#     y_pred = clf.predict(X)
#     accuracy = (y_pred == y).mean()

#     model_path = os.path.join(MODELS_DIR, "f1_lgbm.txt")
#     # Salvar o modelo
#     clf.booster_.save_model(F1_MODEL_PATH)
    
#     print(f"[ML] Modelo f₁ salvo em: {model_path}")    

#     return {
#         "model_path": F1_MODEL_PATH,
#         "accuracy": float(accuracy),
#         "rows_used": len(df),
#         "features": feature_cols,
#     }


# # ============================================================
# #  TRAINER F2 — Neighborhood Model
# # ============================================================

# def train_f2() -> Dict[str, Any]:
#     """
#     Treina LightGBM f₂ (neighborhood ranking).
#     """
#     # if not os.path.exists(F2_DATASET_PATH):
#     #     raise FileNotFoundError(f"Dataset f₂ não encontrado: {F2_DATASET_PATH}")

#     # df = pd.read_csv(F2_DATASET_PATH)
    
#     df = safe_read_csv(F2_DATASET_PATH)

#     if df is None:
#         msg = f"[ML] Dataset f₂ não encontrado em: {F2_DATASET_PATH}\n" \
#             f"→ Gere o dataset antes de treinar (menu ML → Gerar Dataset)."
#         print(msg)
#         return {
#             "success": False,
#             "message": msg,
#             "model_path": None,
#         }

#     print(f"[ML] Treinando modelo f₂... ({len(df)} amostras)")    

#     feature_cols = [
#         "num_periods",
#         "uncovered_need",
#         "avg_load",
#         "load_variance",
#         "heur_total_workers",
#         "opt_total_workers",
#     ]

#     X = df[feature_cols].astype(float)
#     y = df["label"].astype(int)

#     clf = LGBMClassifier(
#         n_estimators=250,
#         learning_rate=0.07,
#         max_depth=-1,
#         num_leaves=48,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         objective="regression_l2"
#     )

#     clf.fit(X, y)

#     y_pred = clf.predict(X)
#     mse = float(np.mean((y - y_pred)**2))

#     model_path = os.path.join(MODELS_DIR, "f2_lgbm.txt")
#     clf.booster_.save_model(F2_MODEL_PATH)
    
#     print(f"[ML] Modelo f₂ salvo em: {model_path}")

#     return {
#         "model_path": F2_MODEL_PATH,
#         "mse": mse,
#         "rows_used": len(df),
#         "features": feature_cols,
#     }


# # ============================================================
# #  TREINAR TUDO EM UM CLIQUE
# # ============================================================

# def train_all_models() -> Dict[str, Any]:
#     """
#     Executa o treinamento de f₁ + f₂ em sequência.
#     """
#     info_f1 = train_f1()
#     info_f2 = train_f2()

#     return {
#         "f1": info_f1,
#         "f2": info_f2,
#     }

# ml/trainer.py  (no seu projeto: simulator/ml/trainer.py)

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    lgb = None
    LGBMClassifier = None
    LGBMRegressor = None

from src.core.i18n import t


# ============================================================
#  PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

F1_DATASET_PATH = os.path.join(DATASETS_DIR, "f1_assignment_dataset.csv")
F2_DATASET_PATH = os.path.join(DATASETS_DIR, "f2_neighborhood_dataset.csv")

F1_MODEL_PATH = os.path.join(MODELS_DIR, "f1_lgbm.txt")
F2_MODEL_PATH = os.path.join(MODELS_DIR, "f2_lgbm.txt")


# ============================================================
#  Helpers de retorno
# ============================================================

def _empty_f1_result(message: str) -> Dict[str, Any]:
    return {
        "success": False,
        "message": message,
        "model_path": None,
        "rows_used": 0,
        "accuracy": None,
    }


def _empty_f2_result(message: str) -> Dict[str, Any]:
    return {
        "success": False,
        "message": message,
        "model_path": None,
        "rows_used": 0,
        "mse": None,
    }


# ============================================================
#  Treino f₁ – Classificador (atribuição motorista × período)
# ============================================================

def train_f1() -> Dict[str, Any]:
    """
    Treina o modelo f₁ usando o CSV f1_assignment_dataset.csv.

    Retorno:
        dict com chaves:
            - success (bool)
            - message (str)
            - model_path (str | None)
            - rows_used (int)
            - accuracy (float | None)
    """
    if lgb is None or LGBMClassifier is None:
        return _empty_f1_result(t("ml_err_lgbm_missing"))

    if not os.path.exists(F1_DATASET_PATH):
        return _empty_f1_result(
            t("ml_err_dataset_missing").format("f1", F1_DATASET_PATH)
        )

    try:
        df = pd.read_csv(F1_DATASET_PATH)
    except Exception as e:
        return _empty_f1_result(t("ml_err_dataset_read").format("f1", e))

    if df.empty:
        return _empty_f1_result(t("ml_err_dataset_empty").format("f1"))

    if "label" not in df.columns:
        return _empty_f1_result(t("ml_err_no_label").format("f1", "label"))

    # features padrão – somente as que existirem
    candidate_features = [
        "driver",
        "period",
        "need",
        "local_load",
        "driver_load",
        "demand_gap",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    if not feature_cols:
        return _empty_f1_result(
            t("ml_err_no_features").format("f1", candidate_features)
        )

    X = df[feature_cols].astype(float)
    y = df["label"].astype(int)

    n = len(df)
    if n < 20:
        # ainda permite treinar, mas avisa
        msg_prefix = t("ml_warn_low_data").format("f1", n)
    else:
        msg_prefix = t("ml_msg_loaded").format("f1", n)

    # Modelo simples LightGBM
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
    )

    try:
        clf.fit(X, y)
    except Exception as e:
        return _empty_f1_result(t("ml_err_train").format("f1", e))

    # Acurácia "no próprio dataset" (não há split train/test nesta fase)
    try:
        y_pred = clf.predict(X)
        accuracy = float((y_pred == y).mean())
    except Exception as e:
        return _empty_f1_result(t("ml_err_acc").format("f1", e))

    # Salvar modelo
    try:
        booster = clf.booster_
        booster.save_model(F1_MODEL_PATH)
    except Exception as e:
        return _empty_f1_result(t("ml_err_save").format("f1", F1_MODEL_PATH, e))

    return {
        "success": True,
        "message": msg_prefix + " " + t("ml_msg_train_success"),
        "model_path": F1_MODEL_PATH,
        "rows_used": int(n),
        "accuracy": accuracy,
    }


# ============================================================
#  Treino f₂ – Regressor (qualidade de vizinhança / melhoria)
# ============================================================

def train_f2() -> Dict[str, Any]:
    """
    Treina o modelo f₂ usando o CSV f2_neighborhood_dataset.csv.

    Retorno:
        dict com chaves:
            - success (bool)
            - message (str)
            - model_path (str | None)
            - rows_used (int)
            - mse (float | None)
    """
    if lgb is None or LGBMRegressor is None:
        return _empty_f2_result(t("ml_err_lgbm_missing"))

    if not os.path.exists(F2_DATASET_PATH):
        return _empty_f2_result(
            t("ml_err_dataset_missing").format("f2", F2_DATASET_PATH)
        )

    try:
        df = pd.read_csv(F2_DATASET_PATH)
    except Exception as e:
        return _empty_f2_result(t("ml_err_dataset_read").format("f2", e))

    if df.empty:
        return _empty_f2_result(t("ml_err_dataset_empty").format("f2"))

    # label padrão → 'label' ou 'improvement'
    target_col = "label" if "label" in df.columns else (
        "improvement" if "improvement" in df.columns else None
    )

    if target_col is None:
        return _empty_f2_result(
            t("ml_err_no_label").format("f2", "label/improvement")
        )

    # features – só as que existirem
    candidate_features = [
        "num_periods",
        "uncovered_need",
        "avg_load",
        "load_variance",
        "heur_total_workers",
        "ref_total_workers",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    if not feature_cols:
        return _empty_f2_result(
            t("ml_err_no_features").format("f2", candidate_features)
        )

    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)

    n = len(df)
    if n < 20:
        msg_prefix = t("ml_warn_low_data").format("f2", n)
    else:
        msg_prefix = t("ml_msg_loaded").format("f2", n)

    # Modelo LightGBM Regressor
    reg = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
    )

    try:
        reg.fit(X, y)
    except Exception as e:
        return _empty_f2_result(t("ml_err_train").format("f2", e))

    # MSE "in-sample"
    try:
        y_pred = reg.predict(X)
        mse = float(((y_pred - y) ** 2).mean())
    except Exception as e:
        return _empty_f2_result(t("ml_err_mse").format("f2", e))

    # Salvar modelo
    try:
        booster = reg.booster_
        booster.save_model(F2_MODEL_PATH)
    except Exception as e:
        return _empty_f2_result(t("ml_err_save").format("f2", F2_MODEL_PATH, e))


    return {
        "success": True,
        "message": msg_prefix + " " + t("ml_msg_train_success"),
        "model_path": F2_MODEL_PATH,
        "rows_used": int(n),
        "mse": mse,
    }


# ============================================================
#  Orquestrador – chamado pela UI (train_all_models)
# ============================================================

def train_all_models() -> Dict[str, Dict[str, Any]]:
    """
    Treina f₁ e f₂ e devolve um dicionário:

        {
            "f1": { ... },
            "f2": { ... },
        }

    Compatível com o bloco da UI no MPythonORToolsV9_3.py
    que faz:

        results = train_all_models()
        f1 = results["f1"]
        f2 = results["f2"]
    """
    f1_info = train_f1()
    f2_info = train_f2()

    return {
        "f1": f1_info,
        "f2": f2_info,
    }
