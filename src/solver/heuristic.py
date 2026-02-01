
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
    cap_tasks_per_driver_per_slot: int,
    assignment_scorer_fn: Optional[Callable[..., Any]] = None
) -> np.ndarray:
    """
    Gera uma alocação inicial baseada em BLOCOS (Janelas Deslizantes).
    Objetivo: Criar turnos contínuos (ex: 4h) para respeitar SHIFT_MIN,
    facilitando o trabalho do LNS/Solver.
    """
    num_periods = len(need)
    allocation = np.zeros((num_periods, limit_workers), dtype=int)
    current_coverage = np.zeros(num_periods, dtype=int)
    
    # Bloco mínimo de 4h30 (18 slots) para alinhar com SHIFT_MIN e regulamento
    BLOCK_SIZE = 18
    actual_len = min(BLOCK_SIZE, num_periods)
    if actual_len <= 0:
        return allocation

    # Para cada motorista, buscamos o melhor bloco disponível
    for driver in range(limit_workers):
        best_start = -1
        best_gain = -1
        
        # Otimização: Passo de 4 slots (1h) para busca rápida
        step = 1
        scores = []
        
        # Varredura (Sliding Window)
        for t in range(0, num_periods - actual_len + 1, step):
            # Ganho = Soma da Demanda Não Atendida neste bloco
            # Se need[k] > current[k], ganhamos 1 ponto.
            gain = 0
            for k in range(t, t + actual_len):
                if need[k] > current_coverage[k]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_start = t
        
        # Se encontrou um bloco útil (ganho > 0), aloca
        if best_start >= 0 and best_gain > 0:
            # Aloca apenas o que é necessário (até a capacidade) para evitar sobre-alocação irreal
            for k in range(best_start, best_start + actual_len):
                needed = max(0, need[k] - current_coverage[k])
                take = min(cap_tasks_per_driver_per_slot, needed)
                allocation[k, driver] = take
                current_coverage[k] += take
            
            # --- TENTATIVA DE EXTENSÃO (Opcional) ---
            # Se sobrar capacidade (em termos de slots livres no turno), estendemos
            # Nota: O regulamento permite turnos maiores, aqui buscamos cobrir demanda residual
            cursor = best_start + actual_len
            total_assigned_in_shift = actual_len 
            
            while total_assigned_in_shift < 36 and cursor < num_periods: # limite arbitrário de 9h para extensão
                needed = max(0, need[cursor] - current_coverage[cursor])
                if needed > 0:
                    take = min(cap_tasks_per_driver_per_slot, needed)
                    allocation[cursor, driver] = take
                    current_coverage[cursor] += take
                    total_assigned_in_shift += 1
                    cursor += 1
                else:
                    break

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
