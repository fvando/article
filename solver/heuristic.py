# import os
# import numpy as np
# from typing import List, Callable, Optional
# # from ml.ml_guidance import f1_predict_score


# # Configuração de caminhos (mantida como no seu original)
# CURRENT_DIR = os.path.dirname(__file__)
# PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# GRANDPARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


# # ============================================================
# # FUNÇÃO UNIVERSAL PARA CHAMADA DO SCORER (LEGACY + ML)
# # ============================================================

# import inspect

# def call_assignment_scorer(
#     assignment_scorer_fn, 
#     driver, 
#     period, 
#     allocation_matrix, 
#     need_vector
# ):
#     if assignment_scorer_fn is None:
#         return None

#     try:
#         sig = inspect.signature(assignment_scorer_fn)
#         params = sig.parameters

#         # tenta por kwargs "clássicas"
#         if {"driver","period","allocation_matrix","need"}.issubset(params):
#             return assignment_scorer_fn(
#                 driver=driver,
#                 period=period,
#                 allocation_matrix=allocation_matrix,
#                 need=need_vector
#             )

#         # tenta por posição (driver, period, allocation_matrix, need)
#         return assignment_scorer_fn(driver, period, allocation_matrix, need_vector)

#     except Exception:
#         # nunca deixa o greedy quebrar
#         return None


# def greedy_initial_allocation(
#     need: List[int],
#     limit_workers: int,
#     max_demands_per_driver: int,
#     assignment_scorer_fn: Optional[Callable] = None
# ) -> np.ndarray:
#     """
#     Gera uma alocação inicial usando heurística clássica ou modelo ML f₁.
#     """

#     num_periods = len(need)
#     allocation = np.zeros((num_periods, limit_workers), dtype=int)

#     sorted_periods = np.argsort(need)[::-1]
#     drivers_demand_count = np.zeros(limit_workers, dtype=int)

#     for period in sorted_periods:
#         required = need[period]
#         scored_workers = []

#         for driver in range(limit_workers):

#             local_load = allocation[period].sum()
#             driver_load = drivers_demand_count[driver]
#             demand_gap = max(0, need[period] - local_load)

#             # -----------------------------
#             # 1️⃣ Tentativa ML (f₁)
#             # -----------------------------
#             # score = None
#             # if assignment_scorer_fn is not None:
#             #     score = assignment_scorer_fn(
#             #         driver=driver,
#             #         period=period,
#             #         allocation_matrix=allocation,
#             #         need=need
#             #     )

#             score = call_assignment_scorer(
#                 assignment_scorer_fn,
#                 driver=driver,
#                 period=period,
#                 allocation_matrix=allocation,
#                 need_vector=need
#             )

#             # -----------------------------
#             # 2️⃣ Fallback heurístico
#             # -----------------------------
#             if score is None:
#                 score = max(0, max_demands_per_driver - driver_load)

#             scored_workers.append((driver, score))

#         # Ordena motoristas por score
#         scored_workers.sort(key=lambda x: x[1], reverse=True)
#         ordered_drivers = [d for d, _ in scored_workers]

#         # Primeira passagem
#         for driver in ordered_drivers:
#             if required <= 0:
#                 break
#             if drivers_demand_count[driver] < max_demands_per_driver:
#                 allocation[period, driver] = 1
#                 drivers_demand_count[driver] += 1
#                 required -= 1

#         # Segunda passagem (fallback)
#         for driver in ordered_drivers:
#             if required <= 0:
#                 break
#             if drivers_demand_count[driver] < max_demands_per_driver:
#                 allocation[period, driver] = 1
#                 drivers_demand_count[driver] += 1
#                 required -= 1

#     return allocation


# # ============================================================
# # (Opcional) Heurística simples para o simulador
# # ============================================================

# def simple_assignment_scorer(driver: int, period: int, allocation: np.ndarray, need: List[int]) -> float:
#     """
#     Heurística simples para guiar alocação:

#     - Penaliza períodos com excesso de motoristas
#     - Prioriza motorista com menos carga acumulada
#     """

#     load_driver = int(allocation[:, driver].sum())
#     load_period = int(allocation[period].sum())

#     return -load_driver - load_period + need[period]


# # def call_assignment_scorer(
# #     assignment_scorer_fn: Optional[Callable],
# #     driver: int,
# #     period: int,
# #     allocation: np.ndarray,
# #     need: List[int]
# # ) -> float:
# #     """
# #     Wrapper que permite dois tipos de scorer:

# #     1) Legacy:
# #         scorer(driver, period, allocation_matrix, need)

# #     2) Machine Learning (LightGBM, etc):
# #         scorer(features_dict)

# #     Retorna sempre um float (escore), mesmo que descartado.
# #     """

# #     if assignment_scorer_fn is None:
# #         return 0.0

# #     # Primeiro tenta estilo ML → recebe dict
# #     try:
# #         features = {
# #             "driver": driver,
# #             "period": period,
# #             "current_load_period": int(allocation[period].sum()),
# #             "need_period": int(need[period]),
# #             "need_gap": int(need[period] - allocation[period].sum()),
# #             "total_allocated": int(allocation.sum()),
# #         }

# #         return float(assignment_scorer_fn(features))

# #     except TypeError:
# #         # Se falhar, assume scorer legacy
# #         try:
# #             return float(assignment_scorer_fn(driver, period, allocation, need))
# #         except Exception:
# #             return 0.0  # devolve neutro para segurança


# # ============================================================
# # HEURÍSTICA GULOSA (BASE)
# # ============================================================

# # def greedy_initial_allocation(
# #     need: List[int],
# #     limit_workers: int,
# #     max_demands_per_driver: int,
# #     assignment_scorer_fn: Optional[Callable] = None
# # ) -> np.ndarray:
# #     """
# #     Gera uma alocação inicial de motoristas usando uma abordagem gulosa.

# #     Parâmetros
# #     ----------
# #     need : list[int]
# #         Vetor de demanda por período.
# #     limit_workers : int
# #         Número máximo de motoristas considerados (colunas da matriz).
# #     max_demands_per_driver : int
# #         Quantidade máxima de períodos que cada motorista pode atender.
# #     assignment_scorer_fn : callable (opcional)
# #         Scorer baseado em ML ou heurística para priorizar motoristas.

# #     Retorno
# #     -------
# #     allocation : np.ndarray (num_periods x limit_workers)
# #         Matriz binária de alocação inicial.
# #     """

# #     num_periods = len(need)
# #     allocation = np.zeros((num_periods, limit_workers), dtype=int)

# #     # Ordena períodos por demanda descrescente
# #     sorted_periods = np.argsort(need)[::-1]

# #     # Contador de quantas demandas cada motorista já atendeu
# #     drivers_demand_count = np.zeros(limit_workers, dtype=int)

# #     for period in sorted_periods:
# #         required = need[period]

# #         # ============================================================
# #         # NOVO: aplica scorer para ordenar motoristas
# #         # ============================================================

# #         scored_workers = []
# #         for driver in range(limit_workers):
            
# #             local_load = allocation[period].sum()
# #             driver_load = drivers_demand_count[driver]
# #             demand_gap = max(0, need[period] - local_load)
            
            
# #             ml_score = None
# #             if f1_model.enabled:
# #                 ml_score = f1_model.predict_assignment_score(
# #                     driver, period, need[period],
# #                     local_load, driver_load, demand_gap
# #                 )

# #             # fallback para heurística clássica
# #             base_score = max(0, max_demands_per_driver - driver_load)

# #             final_score = ml_score if ml_score is not None else base_score
# #             scores.append((final_score, driver))            
            
# #             score = call_assignment_scorer(
# #                 assignment_scorer_fn,
# #                 driver,
# #                 period,
# #                 allocation,
# #                 need
# #             )
# #             scored_workers.append((driver, score))

# #         # Ordena motoristas por score decrescente (melhor primeiro)
# #         scored_workers.sort(key=lambda x: x[1], reverse=True)
# #         ordered_drivers = [drv for drv, _ in scored_workers]

# #         # ============================================================
# #         # PRIMEIRA PASSAGEM: tenta alocar os melhores motoristas
# #         # ============================================================

# #         for driver in ordered_drivers:
# #             if required <= 0:
# #                 break

# #             if drivers_demand_count[driver] < max_demands_per_driver:
# #                 allocation[period, driver] = 1
# #                 drivers_demand_count[driver] += 1
# #                 required -= 1

# #         # ============================================================
# #         # SEGUNDA PASSAGEM: fallback simples
# #         # ============================================================

# #         idx = 0
# #         while required > 0 and idx < limit_workers:
# #             driver = ordered_drivers[idx]
# #             if drivers_demand_count[driver] < max_demands_per_driver:
# #                 allocation[period, driver] = 1
# #                 drivers_demand_count[driver] += 1
# #                 required -= 1
# #             idx += 1

# #     return allocation



# # f1_model = MLGuidanceF1()

# # def greedy_initial_allocation(
# #     need: List[int],
# #     limit_workers: int,
# #     max_demands_per_driver: int,
# #     assignment_scorer_fn: Optional[Callable] = None
# # ) -> np.ndarray:
# #     """
# #     Gera uma alocação inicial usando heurística ou modelo f₁.
# #     """

# #     num_periods = len(need)
# #     allocation = np.zeros((num_periods, limit_workers), dtype=int)

# #     # Ordenar períodos por demanda descrescente
# #     sorted_periods = np.argsort(need)[::-1]

# #     drivers_demand_count = np.zeros(limit_workers, dtype=int)

# #     for period in sorted_periods:
# #         required = need[period]

# #         scored_workers = []

# #         for driver in range(limit_workers):

# #             # features básicas
# #             local_load = allocation[period].sum()
# #             driver_load = drivers_demand_count[driver]
# #             demand_gap = max(0, need[period] - local_load)

# #             # =============================
# #             #  TENTA USAR MODELO f₁
# #             # =============================
# #             # if f1_model.enabled:
# #             #     score = f1_model.predict_assignment_score(
# #             #         driver, period, need[period],
# #             #         local_load, driver_load, demand_gap
# #             #     )
# #             # else:
# #             #     score = None
            

# #             features = {
# #                 "driver": driver,
# #                 "period": period,
# #                 "need_period": need[period],
# #                 "current_load_period": local_load,
# #                 "driver_load": driver_load,
# #                 "need_gap": demand_gap,
# #                 "total_workers": limit_workers,
# #             }

# #             # Se existir modelo → usa ML, senão fallback heurístico
# #             score = f1_predict_score(driver, period, features)
            
            

# #             # =============================
# #             #  FALLBACK HEURÍSTICO
# #             # =============================
# #             if score is None:
# #                 score = max(0, max_demands_per_driver - driver_load)

# #             # =============================
# #             #  SE assignment_scorer_fn EXISTE, SOBRESCREVE
# #             # =============================
# #             if assignment_scorer_fn:
# #                 score = assignment_scorer_fn(
# #                     driver=driver,
# #                     period=period,
# #                     allocation_matrix=allocation,
# #                     need=need
# #                 )

# #             scored_workers.append((driver, score))

# #         # Ordena por score descrescente
# #         scored_workers.sort(key=lambda x: x[1], reverse=True)
# #         ordered_drivers = [d for d, _ in scored_workers]

# #         # =============================
# #         # PRIMEIRA PASSAGEM
# #         # =============================
# #         for driver in ordered_drivers:
# #             if required <= 0:
# #                 break

# #             if drivers_demand_count[driver] < max_demands_per_driver:
# #                 allocation[period, driver] = 1
# #                 drivers_demand_count[driver] += 1
# #                 required -= 1

# #         # =============================
# #         # SEGUNDA PASSAGEM
# #         # =============================
# #         for driver in ordered_drivers:
# #             if required <= 0:
# #                 break

# #             if drivers_demand_count[driver] < max_demands_per_driver:
# #                 allocation[period, driver] = 1
# #                 drivers_demand_count[driver] += 1
# #                 required -= 1

# #     return allocation

# # def greedy_initial_allocation(
# #     need: List[int],
# #     limit_workers: int,
# #     max_demands_per_driver: int,
# #     assignment_scorer_fn: Optional[Callable] = None
# # ) -> np.ndarray:
# #     """
# #     Gera uma alocação inicial usando heurística gulosa,
# #     opcionalmente guiada por ML (f₁).
# #     """

# #     num_periods = len(need)
# #     allocation = np.zeros((num_periods, limit_workers), dtype=int)

# #     # Prioriza períodos mais críticos
# #     sorted_periods = np.argsort(need)[::-1]
# #     drivers_demand_count = np.zeros(limit_workers, dtype=int)

# #     for period in sorted_periods:
# #         required = need[period]
# #         scored_workers = []

# #         for driver in range(limit_workers):

# #             local_load = allocation[period].sum()
# #             driver_load = drivers_demand_count[driver]
# #             demand_gap = max(0, need[period] - local_load)

# #             # ==================================================
# #             # 1) PRIORIDADE MÁXIMA: scorer externo (hook)
# #             # ==================================================
# #             if assignment_scorer_fn is not None:
# #                 score = assignment_scorer_fn(
# #                     driver=driver,
# #                     period=period,
# #                     allocation_matrix=allocation,
# #                     need=need
# #                 )

# #             # ==================================================
# #             # 2) MODELO ML f₁ (se existir)
# #             # ==================================================
# #             else:
# #                 features = {
# #                     "driver": driver,
# #                     "period": period,
# #                     "need_period": need[period],
# #                     "current_load_period": local_load,
# #                     "driver_load": driver_load,
# #                     "need_gap": demand_gap,
# #                     "total_workers": limit_workers,
# #                 }

# #                 score = f1_predict_score(driver, period, features)

# #                 # ==================================================
# #                 # 3) FALLBACK HEURÍSTICO
# #                 # ==================================================
# #                 if score is None:
# #                     score = max(0, max_demands_per_driver - driver_load)

# #             scored_workers.append((driver, score))

# #         # Ordena motoristas por prioridade
# #         scored_workers.sort(key=lambda x: x[1], reverse=True)
# #         ordered_drivers = [d for d, _ in scored_workers]

# #         # Primeira passagem
# #         for driver in ordered_drivers:
# #             if required <= 0:
# #                 break
# #             if drivers_demand_count[driver] < max_demands_per_driver:
# #                 allocation[period, driver] = 1
# #                 drivers_demand_count[driver] += 1
# #                 required -= 1

# #         # Segunda passagem (fallback)
# #         for driver in ordered_drivers:
# #             if required <= 0:
# #                 break
# #             if drivers_demand_count[driver] < max_demands_per_driver:
# #                 allocation[period, driver] = 1
# #                 drivers_demand_count[driver] += 1
# #                 required -= 1

# #     return allocation

from __future__ import annotations

import inspect
from typing import Callable, List, Optional, Any

import numpy as np


# ============================================================
# FUNÇÃO UNIVERSAL PARA CHAMADA DO SCORER (LEGACY + ML)
# ============================================================

def call_assignment_scorer(
    assignment_scorer_fn: Optional[Callable[..., Any]],
    driver: int,
    period: int,
    allocation_matrix: np.ndarray,
    need_vector: List[int],
    *,
    max_demands_per_driver: Optional[int] = None,
    limit_workers: Optional[int] = None,
) -> Optional[float]:
    """
    Chama um scorer de forma resiliente (por kwargs ou por posição),
    suportando versões antigas e novas.

    Retorna:
      - float (score) se conseguir chamar
      - None se não houver scorer ou ocorrer erro
    """
    if assignment_scorer_fn is None:
        return None

    try:
        sig = inspect.signature(assignment_scorer_fn)
        params = sig.parameters

        # Tentativa preferida: kwargs "completas" (nova API do ml_guidance.assignment_scorer)
        if {"driver", "period", "allocation_matrix", "need", "max_demands_per_driver", "limit_workers"}.issubset(params):
            return float(assignment_scorer_fn(
                driver=driver,
                period=period,
                allocation_matrix=allocation_matrix,
                need=need_vector,
                max_demands_per_driver=int(max_demands_per_driver) if max_demands_per_driver is not None else 0,
                limit_workers=int(limit_workers) if limit_workers is not None else allocation_matrix.shape[1],
            ))

        # kwargs "clássicas" (versões antigas)
        if {"driver", "period", "allocation_matrix", "need"}.issubset(params):
            return float(assignment_scorer_fn(
                driver=driver,
                period=period,
                allocation_matrix=allocation_matrix,
                need=need_vector,
            ))

        # fallback posicional
        return float(assignment_scorer_fn(driver, period, allocation_matrix, need_vector))

    except Exception:
        # Nunca deixa o greedy quebrar
        return None


def greedy_initial_allocation(
    need: List[int],
    limit_workers: int,
    max_demands_per_driver: int,
    assignment_scorer_fn: Optional[Callable[..., Any]] = None
) -> np.ndarray:
    """
    Gera uma alocação inicial (num_periods x limit_workers) usando:
      - score ML/externo (se assignment_scorer_fn existir e retornar algo)
      - fallback heurístico (capacidade restante do motorista)

    Observação: esta heurística aloca 0/1 por (period, driver).
    """
    num_periods = len(need)
    allocation = np.zeros((num_periods, limit_workers), dtype=int)

    # Ordenar períodos por demanda descrescente
    sorted_periods = np.argsort(need)[::-1]

    # Quantas vezes cada motorista já foi utilizado (capacidade)
    drivers_demand_count = np.zeros(limit_workers, dtype=int)

    for period in sorted_periods:
        required = int(need[period])
        scored_workers: List[tuple[int, float]] = []

        for driver in range(limit_workers):
            # 1) score via ML/externo (se houver)
            ml_score = call_assignment_scorer(
                assignment_scorer_fn,
                driver=driver,
                period=int(period),
                allocation_matrix=allocation,
                need_vector=need,
                max_demands_per_driver=max_demands_per_driver,
                limit_workers=limit_workers,
            )

            # 2) fallback heurístico (capacidade remanescente)
            base_score = float(max(0, max_demands_per_driver - int(drivers_demand_count[driver])))

            final_score = float(ml_score) if ml_score is not None else base_score
            scored_workers.append((driver, final_score))

        # Ordena por score descrescente
        scored_workers.sort(key=lambda x: x[1], reverse=True)
        ordered_drivers = [d for d, _ in scored_workers]

        # Primeira passagem
        for driver in ordered_drivers:
            if required <= 0:
                break
            if drivers_demand_count[driver] < max_demands_per_driver:
                allocation[period, driver] = 1
                drivers_demand_count[driver] += 1
                required -= 1

        # Segunda passagem (mantida para compatibilidade; hoje tende a ser redundante)
        for driver in ordered_drivers:
            if required <= 0:
                break
            if drivers_demand_count[driver] < max_demands_per_driver:
                allocation[period, driver] = 1
                drivers_demand_count[driver] += 1
                required -= 1

    return allocation


# ============================================================
# Heurística simples opcional (debug / baseline)
# ============================================================

def simple_assignment_scorer(
    driver: int,
    period: int,
    allocation_matrix: np.ndarray,
    need: List[int],
) -> float:
    """
    Heurística simples para guiar alocação:
      - prioriza motorista com menos carga acumulada
      - penaliza período já "carregado"
    """
    load_driver = int(allocation_matrix[:, driver].sum())
    load_period = int(allocation_matrix[period, :].sum())
    return float(-load_driver - load_period + int(need[period]))
