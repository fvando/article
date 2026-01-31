from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


ML_ENABLED = True  # <<< CONTROLE GLOBAL

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

F1_MODEL_PATH = os.path.join(MODELS_DIR, "f1_lgbm.txt")
F2_MODEL_PATH = os.path.join(MODELS_DIR, "f2_lgbm.txt")


# ============================================================
# FEATURE ORDER (NUNCA MUDE SEM RETREINAR)
# ============================================================

F1_FEATURES = [
    # "driver",
    "period",
    "need",
    # "local_load",
    # "driver_load",
    "demand_gap",
]

F2_FEATURES = [
    "num_periods",
    # "uncovered_need",
    "avg_load",
    "load_variance",
    # "heur_total_workers",
    # "opt_total_workers",
]


@dataclass
class MLStatus:
    f1_enabled: bool
    f2_enabled: bool
    f1_path: str
    f2_path: str
    note: str = ""


class MLGuidance:
    """
    Wrapper único do simulador para:
      - f1: scoring motorista×período (greedy)
      - f2: scoring de vizinhança (LNS)

    Se LightGBM ou modelos não existirem, cai em fallback.
    """

    def __init__(self) -> None:
        self.f1_model: Optional[Any] = None
        self.f2_model: Optional[Any] = None

        self.f1_enabled = False
        self.f2_enabled = False

        self._load_models()

    def _load_models(self) -> None:
        if lgb is None:
            # Sem LightGBM instalado → só fallback
            self.f1_model = None
            self.f2_model = None
            self.f1_enabled = False
            self.f2_enabled = False
            return

        # f1
        if os.path.exists(F1_MODEL_PATH):
            try:
                self.f1_model = lgb.Booster(model_file=F1_MODEL_PATH)
                self.f1_enabled = True
            except Exception:
                self.f1_model = None
                self.f1_enabled = False

        # f2
        if os.path.exists(F2_MODEL_PATH):
            try:
                self.f2_model = lgb.Booster(model_file=F2_MODEL_PATH)
                self.f2_enabled = True
            except Exception:
                self.f2_model = None
                self.f2_enabled = False

    def status(self) -> MLStatus:
        note = ""
        if lgb is None:
            note = "lightgbm não instalado → fallback heurístico."
        return MLStatus(
            f1_enabled=self.f1_enabled,
            f2_enabled=self.f2_enabled,
            f1_path=F1_MODEL_PATH,
            f2_path=F2_MODEL_PATH,
            note=note,
        )

    # ---------------------------------------------------------
    # f1
    # ---------------------------------------------------------
    def f1_predict_score(self, features: Dict[str, Any]) -> Optional[float]:
        """
        Retorna prob/score (float) ou None se não houver modelo.
        """
        if not ML_ENABLED:
            return None
        
        if self.f1_model is None:
            return None

        x = np.array([[float(features.get(k, 0.0)) for k in F1_FEATURES]], dtype=float)
        try:
            return float(self.f1_model.predict(x)[0])
        except Exception:
            return None

    def f1_fallback(self, max_demands_per_driver: int, driver_load: int) -> float:
        # score maior para quem ainda tem “capacidade” de pegar demandas
        return float(max(0, max_demands_per_driver - driver_load))

    # ---------------------------------------------------------
    # f2
    # ---------------------------------------------------------
    def f2_predict_score(self, features: Dict[str, Any]) -> Optional[float]:

        if not ML_ENABLED:
            return None
        
        
        if self.f2_model is None:
            return None

        x = np.array([[float(features.get(k, 0.0)) for k in F2_FEATURES]], dtype=float)
        try:
            return float(self.f2_model.predict(x)[0])
        except Exception:
            return None


# Singleton global (importado pelo heuristic/lns)
ml_guidance = MLGuidance()


# ============================================================
# HELPERS (features)
# ============================================================

def extract_f1_features(
    allocation_matrix: np.ndarray,
    need: List[int],
    driver: int,
    period: int,
    total_workers: int,
) -> Dict[str, Any]:
    """
    allocation_matrix: shape (num_periods, num_workers)
    """
    num_periods = allocation_matrix.shape[0]

    if period < 0 or period >= num_periods:
        # período inválido → neutro
        need_p = 0
        local_load = 0
    else:
        need_p = int(need[period]) if period < len(need) else 0
        local_load = int(allocation_matrix[period, :].sum())

    driver_load = int(allocation_matrix[:, driver].sum()) if 0 <= driver < allocation_matrix.shape[1] else 0
    demand_gap = int(max(0, need_p - local_load))

    return {
        "driver": driver,
        "period": period,
        "need": need_p,
        "local_load": local_load,
        "driver_load": driver_load,
        "demand_gap": demand_gap,
        "total_workers": total_workers,
    }


def extract_f2_features(
    current_solution: np.ndarray,
    periods_to_free: List[int],
    need: List[int],
    best_total_workers: int,
) -> Dict[str, Any]:
    """
    Features compatíveis com F2_FEATURES.
    """
    if len(periods_to_free) == 0:
        return {
            "num_periods": 0,
            "uncovered_need": 0,
            "avg_load": 0.0,
            "load_variance": 0.0,
            "heur_total_workers": best_total_workers,
            "opt_total_workers": best_total_workers,
        }

    # load por período (somando motoristas)
    loads = np.array([current_solution[p, :].sum() for p in periods_to_free], dtype=float)

    # uncovered na vizinhança
    uncovered = 0
    for i, p in enumerate(periods_to_free):
        dem = int(need[p]) if p < len(need) else 0
        uncovered += max(0, dem - int(loads[i]))

    return {
        "num_periods": int(len(periods_to_free)),
        "uncovered_need": int(uncovered),
        "avg_load": float(loads.mean()) if loads.size else 0.0,
        "load_variance": float(loads.var()) if loads.size else 0.0,
        "heur_total_workers": int(best_total_workers),
        "opt_total_workers": int(best_total_workers),  # placeholder (pode melhorar depois)
    }


# ============================================================
# WRAPPERS (plug & play)
# ============================================================

def assignment_scorer(
    driver: int,
    period: int,
    allocation_matrix: np.ndarray,
    need: List[int],
    max_demands_per_driver: int,
    limit_workers: int,
) -> float:
    feats = extract_f1_features(
        allocation_matrix=allocation_matrix,
        need=need,
        driver=driver,
        period=period,
        total_workers=limit_workers,
    )

    ml_score = ml_guidance.f1_predict_score(feats)
    if ml_score is not None:
        return ml_score

    return ml_guidance.f1_fallback(max_demands_per_driver=max_demands_per_driver,
                                   driver_load=int(feats["driver_load"]))


def neighborhood_scorer(
    current_solution: np.ndarray,
    periods_to_free: List[int],
    need: List[int],
    best_total_workers: int,
) -> float:
    feats = extract_f2_features(
        current_solution=current_solution,
        periods_to_free=periods_to_free,
        need=need,
        best_total_workers=best_total_workers,
    )

    ml_score = ml_guidance.f2_predict_score(feats)
    if ml_score is not None:
        return ml_score

    # fallback simples: prioriza vizinhanças com mais “uncovered”
    return float(feats["uncovered_need"])
