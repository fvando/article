# ================================================================
# KPIs avançados do simulador Ottimizia
# ================================================================

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

Number = Union[int, float]
VecLike = Union[List[Number], np.ndarray]

def _to_1d_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return None  # int/float NÃO é vetor válido aqui
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        return None
    return arr

# def compute_kpis(demanda, workers_schedule):
#     """Retorna todos os KPIs avançados como um dicionário estruturado."""

#     demanda = np.array(demanda)
#     aloc = np.array(workers_schedule)

#     # Evitar divisão por zero
#     aloc_safe = np.where(aloc == 0, 1e-9, aloc)

#     # --------------------------------------------------------
#     # Cobertura por slot
#     # --------------------------------------------------------
#     coverage = aloc / np.maximum(demanda, 1e-9)
#     abs_diff = np.abs(demanda - aloc)

#     # --------------------------------------------------------
#     # KPIs Globais
#     # --------------------------------------------------------
#     kpis = {}
#     kpis["global_coverage_score"] = 1 - (abs_diff.sum() / demanda.sum())

#     kpis["worker_efficiency"] = np.minimum(aloc, demanda).sum() / aloc.sum()

#     kpis["operational_risk"] = (aloc < demanda).sum() / len(demanda)
#     kpis["operational_risk_severity"] = np.max(demanda - aloc)

#     kpis["overload_index"] = ((aloc - demanda).clip(min=0) / np.maximum(demanda, 1e-9)).mean()
#     kpis["underload_index"] = ((demanda - aloc).clip(min=0) / aloc_safe).mean()

#     # Custo estimado (15 €/hora)
#     custo_motorista = 15 / 4  # 15€/hora → 15-min slot
#     kpis["cost_index"] = aloc.sum() * custo_motorista

#     # --------------------------------------------------------
#     # Temporal Stability
#     # --------------------------------------------------------
#     diffs = np.abs(np.diff(aloc))
#     kpis["temporal_stability"] = 1 - (diffs.mean() / np.maximum(aloc.mean(), 1e-9))

#     # --------------------------------------------------------
#     # Under & Over Coverage
#     # --------------------------------------------------------
#     kpis["total_under_slots"] = (aloc < demanda).sum()
#     kpis["total_over_slots"] = (aloc > demanda).sum()

#     # --------------------------------------------------------
#     # Safety Margin por slot
#     # --------------------------------------------------------
#     safety = aloc - demanda
#     kpis["safety_margin_mean"] = safety.mean()
#     kpis["safety_margin_min"] = safety.min()
#     kpis["safety_margin_max"] = safety.max()

#     return kpis, coverage, safety

# import numpy as np
# from typing import Any, Dict, Tuple


# # def compute_kpis(
# #     demanda,
# #     workers_schedule
# # ) -> Tuple[Dict[str, Any], float, float]:
# #     """
# #     Calcula KPIs operacionais a partir do schedule.

# #     Se workers_schedule for None ou inválido,
# #     retorna KPIs neutros sem quebrar o fluxo.
# #     """

# #     # =====================================================
# #     # CASO 1 — Não existe schedule (heuristic / lns)
# #     # =====================================================
# #     if workers_schedule is None:
# #         # kpis = {
# #         #     "num_switches": None,
# #         #     "avg_shift_length": None,
# #         #     "max_consecutive_work": None,
# #         #     "global_coverage_score": None,   # 👈 alias
# #         #     "note": "KPIs completos indisponíveis (workers_schedule ausente)"
# #         # }
        
# #         KPIS = {
# #             # =========================
# #             # KPIs SLOT-LEVEL (novos)
# #             # =========================
# #             "coverage": np.ndarray | None,
# #             "safety_margin": np.ndarray | None,

# #             # =========================
# #             # KPIs AGREGADOS (novos)
# #             # =========================
# #             "global_coverage": float | None,
# #             "worker_efficiency": float | None,
# #             "operational_risk": float | None,
# #             "temporal_stability": float | None,
# #             "cost_index": float | None,

# #             # =========================
# #             # KPIs LEGADOS (mantidos)
# #             # =========================
# #             "global_coverage_score": float | None,   # 👈 alias
# #             "num_switches": int | None,
# #             "avg_shift_length": float | None,
# #             "max_consecutive_work": int | None,

# #             # =========================
# #             # Metadata
# #             # =========================
# #             "note": str | None
# #         }


        
# #         coverage = float("nan")
# #         safety_margin = float("nan")
# #         return kpis, coverage, safety_margin

# #     # =====================================================
# #     # CASO 2 — Garantir array NumPy válido
# #     # =====================================================
# #     aloc = np.asarray(workers_schedule)

# #     if aloc.ndim == 0 or aloc.size == 0:
# #         # kpis = {
# #         #     "num_switches": None,
# #         #     "avg_shift_length": None,
# #         #     "max_consecutive_work": None,
# #         #     "global_coverage_score": None,   # 👈 alias
# #         #     "note": "KPIs indisponíveis (schedule inválido)"
# #         # }
        
# #         KPIS = {
# #             # =========================
# #             # KPIs SLOT-LEVEL (novos)
# #             # =========================
# #             "coverage": np.ndarray | None,
# #             "safety_margin": np.ndarray | None,

# #             # =========================
# #             # KPIs AGREGADOS (novos)
# #             # =========================
# #             "global_coverage": float | None,
# #             "worker_efficiency": float | None,
# #             "operational_risk": float | None,
# #             "temporal_stability": float | None,
# #             "cost_index": float | None,

# #             # =========================
# #             # KPIs LEGADOS (mantidos)
# #             # =========================
# #             "global_coverage_score": float | None,   # 👈 alias
# #             "num_switches": int | None,
# #             "avg_shift_length": float | None,
# #             "max_consecutive_work": int | None,

# #             # =========================
# #             # Metadata
# #             # =========================
# #             "note": str | None
# #         }
        
        
        
# #         coverage = float("nan")
# #         safety_margin = float("nan")
# #         return kpis, coverage, safety_margin

# #     # =====================================================
# #     # CASO 3 — KPIs reais (modo Exact)
# #     # =====================================================

# #     # Se vier 2D (periods × drivers), agregamos por período
# #     if aloc.ndim == 2:
# #         aloc_1d = aloc.sum(axis=1)
# #     else:
# #         aloc_1d = aloc

# #     # Trocas (on/off)
# #     diffs = np.abs(np.diff(aloc_1d))
# #     num_switches = int(diffs.sum())

# #     # Comprimento médio de turnos
# #     work_periods = aloc_1d > 0
# #     avg_shift_length = float(work_periods.sum() / max(1, np.count_nonzero(diffs)))

# #     # Máximo de trabalho consecutivo
# #     max_consecutive = 0
# #     current = 0
# #     for v in work_periods:
# #         if v:
# #             current += 1
# #             max_consecutive = max(max_consecutive, current)
# #         else:
# #             current = 0

# #     # Cobertura simples
# #     demanda = np.asarray(demanda)
# #     coverage = float(
# #         min(1.0, aloc_1d.sum() / max(1.0, demanda.sum()))
# #     )

# #     # Margem de segurança (simples)
# #     safety_margin = float(np.mean(demanda - aloc_1d))

# #     # kpis = {
# #     #     "num_switches": num_switches,
# #     #     "avg_shift_length": avg_shift_length,
# #     #     "max_consecutive_work": max_consecutive,
# #     #     "global_coverage_score": coverage,   # 👈 alias
# #     #     "note": "KPIs completos calculados"
# #     # }
    
# #     kpis = {
# #         # ==================================================
# #         # KPIs LEGADOS (mantidos – NÃO REMOVER)
# #         # ==================================================
# #         "num_switches": num_switches,
# #         "avg_shift_length": avg_shift_length,
# #         "max_consecutive_work": max_consecutive,

# #         # alias histórico (UI antiga / dissertação)
# #         "global_coverage_score": global_coverage,

# #         # ==================================================
# #         # KPIs NOVOS – AGREGADOS
# #         # ==================================================
# #         "global_coverage": global_coverage,
# #         "worker_efficiency": worker_efficiency,
# #         "operational_risk": operational_risk,
# #         "temporal_stability": temporal_stability,
# #         "cost_index": cost_index,

# #         # ==================================================
# #         # KPIs NOVOS – SLOT-LEVEL (para gráficos)
# #         # ==================================================
# #         "coverage": coverage_vector,          # np.ndarray ou None
# #         "safety_margin": safety_margin_vector, # np.ndarray ou None

# #         # ==================================================
# #         # Metadata
# #         # ==================================================
# #         "note": "KPIs completos calculados"
# #     }
    

# #     return kpis, coverage, safety_margin

# def compute_kpis(demanda, workers_schedule):
#     """
#     Computa KPIs do simulador.

#     Retorna:
#     - kpis: dict com contrato estável
#     - coverage: None (placeholder)
#     - safety_margin: None (placeholder)
#     """

#     # # --------------------------------------------------
#     # # KPIs LEGADOS (os únicos realmente calculados hoje)
#     # # --------------------------------------------------
#     # if workers_schedule is None or len(workers_schedule) == 0:
#     #     kpis = {
#     #         "num_switches": None,
#     #         "avg_shift_length": None,
#     #         "max_consecutive_work": None,
#     #         "global_coverage_score": None,
#     #         "note": "KPIs indisponíveis (workers_schedule ausente)"
#     #     }
#     #     return kpis, None, None

#     if not isinstance(workers_schedule, (list, tuple, np.ndarray)):
#         kpis = {
#             "num_switches": None,
#             "avg_shift_length": None,
#             "max_consecutive_work": None,
#             "global_coverage_score": None,
#             "note": "KPIs indisponíveis (workers_schedule não vetorial)"
#         }

#         # placeholders do contrato novo
#         kpis.update({
#             "global_coverage": None,
#             "worker_efficiency": None,
#             "operational_risk": None,
#             "temporal_stability": None,
#             "cost_index": None,
#             "coverage": None,
#             "safety_margin": None,
#         })

#         return kpis, None, None


#     # Exemplo simples (ajuste se já tiver lógica melhor)
#     workers = np.array(workers_schedule)
#     demanda = np.array(demanda)

#     coverage_scalar = np.mean(
#         np.minimum(workers, demanda) / np.maximum(demanda, 1)
#     )

#     # KPIs legados efetivos
#     kpis = {
#         "num_switches": None,                 # ainda não implementado
#         "avg_shift_length": None,             # ainda não implementado
#         "max_consecutive_work": None,         # ainda não implementado
#         "global_coverage_score": float(coverage_scalar),
#         "note": "KPIs básicos calculados"
#     }

#     # --------------------------------------------------
#     # Placeholders explícitos (NOVO CONTRATO)
#     # --------------------------------------------------
#     kpis.update({
#         # agregados futuros
#         "global_coverage": None,
#         "worker_efficiency": None,
#         "operational_risk": None,
#         "temporal_stability": None,
#         "cost_index": None,

#         # vetoriais futuros (para gráficos)
#         "coverage": None,
#         "safety_margin": None,
#     })

#     return kpis, None, None

# solver/kpis.py
# from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


ArrayLike = Union[List[float], List[int], np.ndarray]


def _to_1d_numeric(x: Any) -> Optional[np.ndarray]:
    """
    Converte x para np.ndarray 1D numérico, ou retorna None se não for possível.
    """
    if x is None:
        return None

    # int/float escalar -> vetor 1D de tamanho 1
    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.array([float(x)], dtype=float)

    # list/tuple/ndarray
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return np.array([float(arr)], dtype=float)
        if arr.ndim > 1:
            # tenta achatar sem explodir
            arr = arr.reshape(-1)
        return arr

    return None


# def compute_kpis(
#     demanda: ArrayLike,
#     workers_schedule: Any
# ) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
#     """
#     KPIs canônicos e defensivos.

#     Entrada esperada:
#       - demanda: vetor (num_periods,)
#       - workers_schedule: vetor (num_periods,) OU None.
#         (se vier int/float/obj, a função não quebra e devolve KPIs None)

#     Saída:
#       - kpis: dict com contrato estável
#       - coverage_vector: np.ndarray (num_periods,) ou None
#       - safety_margin_vector: np.ndarray (num_periods,) ou None
#     """

#     # ------------------------------------------------------------
#     # Contrato estável (para a UI nunca quebrar por KeyError)
#     # ------------------------------------------------------------
#     kpis: Dict[str, Any] = {
#         # agregados (UI)
#         "global_coverage_score": None,
#         "worker_efficiency": None,

#         # vetoriais (gráficos)
#         "coverage": None,
#         "safety_margin": None,

#         # extras (mantém compatibilidade com UI antiga / placeholders)
#         "operational_risk": None,
#         "operational_risk_severity": None,
#         "temporal_stability": None,
#         "cost_index": None,

#         # legados (se existirem no teu UI antigo)
#         "num_switches": None,
#         "avg_shift_length": None,
#         "max_consecutive_work": None,

#         "note": "",
#     }

#     demand_vec = _to_1d_numeric(demanda)
#     work_vec = _to_1d_numeric(workers_schedule)

#     if demand_vec is None or demand_vec.size == 0:
#         kpis["note"] = "KPIs indisponíveis (demanda inválida ou vazia)"
#         return kpis, None, None

#     if work_vec is None:
#         kpis["note"] = "KPIs indisponíveis (workers_schedule inválido / não vetorial)"
#         return kpis, None, None

#     # Alinhar comprimentos: se o teu workers_schedule vier com tamanho diferente, não quebrar.
#     if demand_vec.size != work_vec.size:
#         kpis["note"] = f"Dimensões incompatíveis (demanda={demand_vec.size}, workers={work_vec.size})"
#         return kpis, None, None

#     # ------------------------------------------------------------
#     # Vetores base (para gráficos)
#     # ------------------------------------------------------------
#     # coverage por slot: min(workers, demand)/max(demand,1)
#     denom = np.maximum(demand_vec, 1.0)
#     coverage_vec = np.minimum(work_vec, demand_vec) / denom

#     # margem de segurança por slot: workers - demand
#     safety_margin_vec = work_vec - demand_vec

#     # ------------------------------------------------------------
#     # Agregados
#     # ------------------------------------------------------------
#     global_coverage = float(np.nanmean(coverage_vec)) if coverage_vec.size else None

#     total_demand = float(np.nansum(demand_vec))
#     total_workers = float(np.nansum(work_vec))

#     # eficiência simples: quanto “custo” em workers para atender demanda
#     worker_eff = None
#     if total_demand > 0:
#         worker_eff = float(total_workers / total_demand)

#     # ------------------------------------------------------------
#     # Preencher contrato
#     # ------------------------------------------------------------
#     kpis["coverage"] = coverage_vec
#     kpis["worker_efficiency"] = worker_eff
#     kpis["safety_margin"] = safety_margin_vec
#     kpis["global_coverage_score"] = global_coverage
#     kpis["note"] = "KPIs básicos calculados"

#     return kpis, coverage_vec, safety_margin_vec

def calculate_gini(values: np.ndarray) -> float:
    """
    Calcula o Índice de Gini para um vetor de valores (ex: horas por motorista).
    0 = igualdade perfeita, 1 = desigualdade máxima.
    """
    if values is None or values.size == 0:
        return 0.0
    
    # Valores devem ser não negativos e ordenados
    x = np.sort(values.astype(float))
    n = len(x)
    
    if np.sum(x) <= 0:
        return 0.0
        
    # Lorenz Curve approach
    cumsum_x = np.cumsum(x)
    cumsum_x = np.insert(cumsum_x, 0, 0)
    lorenz = cumsum_x / cumsum_x[-1]
    
    # Gini = 1 - 2 * area under Lorenz curve
    # Area under Lorenz curve using trapezoidal rule
    area = np.trapz(lorenz, np.linspace(0, 1, n + 1))
    return float(1 - 2 * area)

def compute_kpis(
    demanda: VecLike,
    workers_schedule: Any = None,
    matrix_allocation: Optional[np.ndarray] = None,
    cost_per_worker_slot: float = 1.0,
) -> Tuple[Dict[str, Any], Optional[float], Optional[np.ndarray]]:
    """
    Retorna:
      - kpis: dict com chaves fixas (para UI não quebrar)
      - global_coverage_score: float (alias histórico)
      - safety_margin: vetor (slots)
    """

    d = _to_1d_array(demanda)

    # Fonte primária: matrix_allocation
    if isinstance(matrix_allocation, np.ndarray) and matrix_allocation.ndim == 2:
        w = matrix_allocation.sum(axis=1)
        # Horas por motorista (para Gini)
        # Assumindo que cada slot = 0.25h (15min)
        driving_hours = matrix_allocation.sum(axis=0) * 0.25
        active_hours = driving_hours[driving_hours > 0]
        gini_index = calculate_gini(active_hours)
    else:
        w = _to_1d_array(workers_schedule)
        gini_index = None
    

    # contrato fixo (NUNCA remove chaves)
    kpis: Dict[str, Any] = {
        # antigos / compatibilidade
        "num_switches": None,
        "avg_shift_length": None,
        "max_consecutive_work": None,
        "global_coverage_score": None,

        # novos agregados
        "global_coverage": None,
        "worker_efficiency": None,
        "operational_risk": None,
        "operational_risk_severity": None,
        "temporal_stability": None,
        "cost_index": None,
        "gini_index": gini_index,

        # vetores para gráficos
        "coverage": None,
        "safety_margin": None,

        "note": "",
    }

    if d is None or d.size == 0:
        kpis["note"] = "Demanda inválida/ausente."
        return kpis, None, None

    if w is None or w.size == 0:
        kpis["note"] = "Dados insuficientes para KPIs slot-level."
        return kpis, None, None    

    if w.size != d.size:
        kpis["note"] = f"Tamanhos incompatíveis: demanda={d.size} vs workers_schedule={w.size}."
        return kpis, None, None

    # -------- vetores base --------
    safety_margin = w - d

    # coverage por slot (evita div/0: se demanda==0, definimos coverage=1)
    coverage_vec = np.ones_like(d, dtype=float)
    mask = d > 0
    coverage_vec[mask] = w[mask] / d[mask]
    coverage_vec = np.clip(coverage_vec, 0.0, 2.0)  # clip só pra evitar outliers

    # -------- agregados --------
    total_demand = float(np.sum(d))
    total_workers_slots = float(np.sum(w))

    # cobertura global ponderada por demanda (mais estável para dissertação)
    if total_demand > 0:
        global_coverage = float(np.sum(np.minimum(w, d)) / total_demand)
        under = float(np.sum(np.maximum(d - w, 0.0)) / total_demand)
    else:
        global_coverage = 1.0
        under = 0.0

    # eficiência: “quanto do que eu aloco é realmente necessário”
    # (1.0 = perfeito, menor = desperdício)
    if total_workers_slots > 0:
        worker_eff = float(np.sum(np.minimum(w, d)) / total_workers_slots)
    else:
        worker_eff = None

    # estabilidade temporal: menos “serrilhado” no perfil, melhor
    diffs = np.abs(np.diff(w))
    num_switches = int(np.sum(diffs > 0))
    temporal_stability = float(1.0 / (1.0 + np.mean(diffs))) if diffs.size > 0 else 1.0

    # custo (índice simples)
    cost_index = float(total_workers_slots * float(cost_per_worker_slot))

    # severidade do risco (1..5)
    # under = fração de demanda não coberta
    if under <= 0.01:
        sev = 1
    elif under <= 0.05:
        sev = 2
    elif under <= 0.10:
        sev = 3
    elif under <= 0.20:
        sev = 4
    else:
        sev = 5

    # preencher contrato
    kpis.update({
        "num_switches": num_switches,
        "avg_shift_length": None,         # você pode calcular depois se tiver turnos individuais
        "max_consecutive_work": None,     # idem
        "global_coverage_score": global_coverage,  # alias histórico
        "global_coverage": global_coverage,
        "worker_efficiency": worker_eff,
        "operational_risk": under,
        "operational_risk_severity": sev,
        "temporal_stability": temporal_stability,
        "cost_index": cost_index,
        "coverage": coverage_vec,
        "safety_margin": safety_margin,
        "gini_index": gini_index,
        "note": "KPIs completos calculados",
    })

    return kpis, global_coverage, safety_margin
