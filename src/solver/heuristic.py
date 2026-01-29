
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
    
    
    # Ordena por demanda, mas com leve viés para atender slots mais tarde também
    # T = max(1, num_periods - 1)
    # lambda_time = 0.25  # ajuste pequeno; 0.1–0.5 costuma bastar
    # scores = np.array([need[p] + lambda_time * (p / T) for p in range(num_periods)], dtype=float)
    # sorted_periods = np.argsort(scores)[::-1]


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
