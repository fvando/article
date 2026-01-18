# import os
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import pandas as pd


# # ============================================================
# #  Diretórios / caminhos padrão
# # ============================================================

# BASE_DIR = os.path.dirname(__file__)
# DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")

# os.makedirs(DATASETS_DIR, exist_ok=True)

# F1_DATASET_PATH = os.path.join(DATASETS_DIR, "f1_assignment_dataset.csv")
# F2_DATASET_PATH = os.path.join(DATASETS_DIR, "f2_neighborhood_dataset.csv")


# # ============================================================
# #  BUILD: LINHAS DO DATASET f1 (ASSIGNMENT MODEL)
# # ============================================================

# def build_f1_rows(
#     instance_id: int,
#     need: List[int],
#     allocation_optimal: np.ndarray,
# ) -> List[Dict[str, Any]]:
#     """
#     Gera linhas para o dataset f1 a partir de uma solução ótima (ou de alta qualidade).

#     Cada linha representa um par (driver, period) com features e rótulo.

#     - instance_id: identificador da instância sintética
#     - need: vetor de demanda por período
#     - allocation_optimal: matriz (num_periods x num_drivers) com solução ótima
#     """
#     num_periods, num_workers = allocation_optimal.shape
#     rows: List[Dict[str, Any]] = []

#     # Pré-cálculos
#     allocation_optimal = allocation_optimal.astype(int)
#     load_per_period = allocation_optimal.sum(axis=1)   # shape: (num_periods,)
#     load_per_driver = allocation_optimal.sum(axis=0)   # shape: (num_workers,)

#     for p in range(num_periods):
#         need_p = int(need[p])
#         local_load = int(load_per_period[p])
#         demand_gap = int(need_p - local_load)

#         for d in range(num_workers):
#             assigned_opt = int(allocation_optimal[p, d])
#             driver_load = int(load_per_driver[d])

#             row = {
#                 "instance_id": instance_id,
#                 "driver": d,
#                 "period": p,
#                 "need": need_p,
#                 "local_load": local_load,
#                 "driver_load": driver_load,
#                 "demand_gap": demand_gap,
#                 # rótulo: se (d,p) está ligado na solução ótima
#                 "assigned_optimal": assigned_opt,
#                 "label": assigned_opt,
#             }
#             rows.append(row)

#     return rows


# # ============================================================
# #  BUILD: LINHAS DO DATASET f2 (NEIGHBORHOOD MODEL)
# # ============================================================

# def build_f2_rows(
#     instance_id: int,
#     need: List[int],
#     allocation_optimal: np.ndarray,
#     total_workers_greedy: int,
#     total_workers_optimal: int,
#     neighborhood_id: int = 0,
#     periods: List[int] | None = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Gera linhas para o dataset f2 a partir de:

#     - solução ótima (allocation_optimal)
#     - solução heurística (via total_workers_greedy)
#     - lista de períodos da vizinhança (por padrão, todos)

#     Aqui usamos uma única vizinhança por instância (neighborhood_id=0 por padrão),
#     mas é fácil expandir depois para múltiplas vizinhanças.
#     """
#     if periods is None:
#         periods = list(range(len(need)))

#     allocation_optimal = allocation_optimal.astype(int)

#     # Carga por período apenas dentro da vizinhança
#     local_loads = np.array(
#         [allocation_optimal[p, :].sum() for p in periods],
#         dtype=int,
#     )

#     uncovered_need = int(
#         sum(max(0, int(need[p]) - int(local_loads[i])) for i, p in enumerate(periods))
#     )

#     avg_load = float(local_loads.mean()) if len(local_loads) > 0 else 0.0
#     load_variance = float(local_loads.var()) if len(local_loads) > 0 else 0.0

#     improvement = int(max(0, total_workers_greedy - total_workers_optimal))

#     row = {
#         "instance_id": instance_id,
#         "neighborhood_id": neighborhood_id,
#         "num_periods": len(periods),
#         "uncovered_need": uncovered_need,
#         "avg_load": avg_load,
#         "load_variance": load_variance,
#         "heur_total_workers": int(total_workers_greedy),
#         "opt_total_workers": int(total_workers_optimal),
#         "improvement": improvement,
#         "label": improvement,
#     }

#     return [row]


# # ============================================================
# #  SALVAR DATASETS EM CSV (APPEND SE EXISTIR)
# # ============================================================

# def _append_or_write(df: pd.DataFrame, path: str) -> None:
#     """
#     Salva DataFrame em CSV:
#     - se o arquivo não existe: cria com header
#     - se já existe: faz append sem header
#     """
#     if df.empty:
#         return

#     file_exists = os.path.exists(path)
#     df.to_csv(path, mode="a" if file_exists else "w", index=False, header=not file_exists)


# def save_datasets(
#     f1_rows: List[Dict[str, Any]],
#     f2_rows: List[Dict[str, Any]],
#     f1_path: str = F1_DATASET_PATH,
#     f2_path: str = F2_DATASET_PATH,
# ) -> Tuple[str, str]:
#     """
#     Converte listas de dicionários em DataFrames e salva nos CSVs f1 e f2.

#     Retorna os caminhos finais usados.
#     """
#     df_f1 = pd.DataFrame(f1_rows) if f1_rows else pd.DataFrame()
#     df_f2 = pd.DataFrame(f2_rows) if f2_rows else pd.DataFrame()

#     _append_or_write(df_f1, f1_path)
#     _append_or_write(df_f2, f2_path)

#     return f1_path, f2_path


# ============================================================
#  dataset_builder.py – versão flexível para método fonte
#  Autor: Vando Moreira (Simulador Acadêmico)
# ============================================================

# import os
# from typing import Any, Dict, List, Tuple, Optional

# import numpy as np
# import pandas as pd


# # ============================================================
# #  Diretórios padrão
# # ============================================================

# BASE_DIR = os.path.dirname(__file__)
# DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
# os.makedirs(DATASETS_DIR, exist_ok=True)

# F1_DATASET_PATH = os.path.join(DATASETS_DIR, "f1_assignment_dataset.csv")
# F2_DATASET_PATH = os.path.join(DATASETS_DIR, "f2_neighborhood_dataset.csv")


# # ============================================================
# #  FUNÇÃO AUXILIAR PARA SELECIONAR O GROUND TRUTH FLEXÍVEL
# # ============================================================

# def choose_ground_truth(
#     need: List[int],
#     solve_exact_fn,
#     solve_heuristic_fn,
#     solve_lns_fn,
#     source_preference: str = "auto",
#     instance_id: int = 0,
#     verbose: bool = False,
# ) -> Tuple[np.ndarray, str, int]:
#     """
#     Seleciona a solução 'ótima' flexível para geração do dataset.

#     Parâmetros
#     ----------
#     need : lista de demandas
#     solve_exact_fn : função chamando solver exato
#     solve_heuristic_fn : função chamando heurística gulosa
#     solve_lns_fn : função chamando LNS
#     source_preference : {"auto", "exact", "heuristic", "lns"}
#     instance_id : rótulo da instância (usado no dataset)
#     verbose : log detalhado

#     Retorna
#     -------
#     allocation_ground_truth : matriz binária (periodos × motoristas)
#     source_method : string indicando qual método gerou a solução
#     total_workers_gt : soma binária da solução
#     """

#     # -----------------------------------------
#     # 1) Modo manual (força um método específico)
#     # -----------------------------------------
#     if source_preference == "exact":
#         try:
#             sol, tw = solve_exact_fn(need)
#             return sol, "exact", tw
#         except Exception:
#             raise RuntimeError("Exact solver falhou e source_preference=exact não permite fallback.")

#     if source_preference == "heuristic":
#         sol, tw = solve_heuristic_fn(need)
#         return sol, "heuristic", tw

#     if source_preference == "lns":
#         sol, tw = solve_lns_fn(need)
#         return sol, "lns", tw

#     # -----------------------------------------
#     # 2) AUTO – escolhe o melhor disponível
#     # -----------------------------------------
#     assert source_preference == "auto"

#     # Tenta EXACT
#     try:
#         sol, tw = solve_exact_fn(need)
#         if verbose:
#             print(f"[GT] Exact solver OK para instância {instance_id}")
#         return sol, "exact", tw
#     except Exception:
#         if verbose:
#             print(f"[GT] Exact falhou → tentando LNS para instância {instance_id}")

#     # Tenta LNS
#     try:
#         sol, tw = solve_lns_fn(need)
#         if verbose:
#             print(f"[GT] LNS OK para instância {instance_id}")
#         return sol, "lns", tw
#     except Exception:
#         if verbose:
#             print(f"[GT] LNS falhou → usando heurística para instância {instance_id}")

#     # Fallback → Heurística
#     sol, tw = solve_heuristic_fn(need)
#     return sol, "heuristic", tw
    


# # ============================================================
# #  BUILD f₁ – Assignment-level dataset
# # ============================================================

# def build_f1_rows(
#     instance_id: int,
#     need: List[int],
#     allocation_gt: np.ndarray,
#     source_method: str,
# ) -> List[Dict[str, Any]]:

#     num_periods, num_workers = allocation_gt.shape
#     rows = []

#     allocation_gt = allocation_gt.astype(int)
#     load_period = allocation_gt.sum(axis=1)
#     load_driver = allocation_gt.sum(axis=0)

#     for p in range(num_periods):
#         for d in range(num_workers):

#             assigned = int(allocation_gt[p, d])
#             demand_gap = int(need[p] - load_period[p])

#             rows.append({
#                 "instance_id": instance_id,
#                 "source_method": source_method,
#                 "period": p,
#                 "driver": d,
#                 "need": int(need[p]),
#                 "local_load": int(load_period[p]),
#                 "driver_load": int(load_driver[d]),
#                 "demand_gap": demand_gap,
#                 "label": assigned,         # ground truth
#             })

#     return rows



# # ============================================================
# #  BUILD f₂ – Neighborhood-level dataset
# # ============================================================

# def build_f2_rows(
#     instance_id: int,
#     need: List[int],
#     allocation_gt: np.ndarray,
#     total_workers_greedy: int,
#     total_workers_gt: int,
#     source_method: str,
#     neighborhood_id: int = 0,
#     periods: Optional[List[int]] = None,
# ) -> List[Dict[str, Any]]:

#     if periods is None:
#         periods = list(range(len(need)))

#     allocation_gt = allocation_gt.astype(int)

#     # Calcula cargas dentro da vizinhança
#     local_loads = np.array([allocation_gt[p].sum() for p in periods])
#     uncovered = int(sum(max(0, need[p] - local_loads[i]) for i, p in enumerate(periods)))

#     avg_load = float(local_loads.mean()) if len(local_loads) else 0.0
#     load_var = float(local_loads.var()) if len(local_loads) else 0.0

#     improvement = int(max(0, total_workers_greedy - total_workers_gt))

#     return [{
#         "instance_id": instance_id,
#         "source_method": source_method,
#         "neighborhood_id": neighborhood_id,
#         "num_periods": len(periods),
#         "uncovered_need": uncovered,
#         "avg_load": avg_load,
#         "load_variance": load_var,
#         "heur_total_workers": int(total_workers_greedy),
#         "gt_total_workers": int(total_workers_gt),
#         "improvement": improvement,
#         "label": improvement,
#     }]



# # ============================================================
# #  SALVAR DATASETS EM CSV
# # ============================================================

# def save_datasets(f1_rows, f2_rows) -> Tuple[str, str]:
#     df1 = pd.DataFrame(f1_rows)
#     df2 = pd.DataFrame(f2_rows)

#     # append resiliente
#     write_opt(df1, F1_DATASET_PATH)
#     write_opt(df2, F2_DATASET_PATH)

#     return F1_DATASET_PATH, F2_DATASET_PATH


# def write_opt(df: pd.DataFrame, path: str):
#     if df.empty:
#         return

#     exists = os.path.exists(path) and os.path.getsize(path) > 10
#     df.to_csv(path, mode="a" if exists else "w", index=False, header=not exists)


import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

F1_DATASET_PATH = os.path.join(DATASETS_DIR, "f1_assignment_dataset.csv")
F2_DATASET_PATH = os.path.join(DATASETS_DIR, "f2_neighborhood_dataset.csv")


def _ensure_matrix_and_schedule(
    need: np.ndarray,
    workers_schedule: list[int],
    matrix_allocation: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    Garante que:
    - matrix_allocation é 2D (period x workers)
    - workers_schedule é coerente com matrix
    """
    if matrix_allocation is None:
        raise ValueError("matrix_allocation não pode ser None no dataset_builder")

    if matrix_allocation.ndim != 2:
        raise ValueError("matrix_allocation deve ser 2D")

    ws = list(np.sum(matrix_allocation, axis=1).astype(int))

    if len(ws) != len(need):
        raise ValueError("workers_schedule incompatível com need")

    return matrix_allocation, ws


# ============================================================
#   AUXILIAR — WRAPPERS UNIFORMES PARA OS SOLVERS
# ============================================================

def unpack_solver_result(result):
    """
    Aceita o retorno do solver exato / heurístico / LNS
    e extrai somente (matrix_allocation, total_workers).
    """
    (
        _solver,
        _status,
        total_workers,
        _workers_schedule,
        _constraints,
        _dens1,
        _dens2,
        _stats,
        _msg,
        _iters,
        matrix,
        _logs,
    ) = result

    if matrix is None:
        raise RuntimeError("Solver retornou matrix_allocation=None")

    return matrix.astype(int), int(matrix.sum())


# ============================================================
#  SELECTOR FLEXÍVEL DO GROUND TRUTH
# ============================================================

def choose_ground_truth(
    need: List[int],
    solve_exact_fn,
    solve_heuristic_fn,
    solve_lns_fn,
    source_preference: str = "auto",
    instance_id: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, str, int]:

    if source_preference == "exact":
        sol, tw = unpack_solver_result(solve_exact_fn(need))
        return sol, "exact", tw

    if source_preference == "heuristic":
        sol, tw = unpack_solver_result(solve_heuristic_fn(need))
        return sol, "heuristic", tw

    if source_preference == "lns":
        sol, tw = unpack_solver_result(solve_lns_fn(need))
        return sol, "lns", tw

    # AUTO
    try:
        sol, tw = unpack_solver_result(solve_exact_fn(need))
        return sol, "exact", tw
    except Exception:
        if verbose:
            print(f"[GT] Exact falhou → LNS instância {instance_id}")

    try:
        sol, tw = unpack_solver_result(solve_lns_fn(need))
        return sol, "lns", tw
    except Exception:
        if verbose:
            print(f"[GT] LNS falhou → Heuristic instância {instance_id}")

    sol, tw = unpack_solver_result(solve_heuristic_fn(need))
    return sol, "heuristic", tw


# ============================================================
#  BUILD f₁ — Assignment-level dataset
# ============================================================

# def build_f1_rows(
#     instance_id: int,
#     need: List[int],
#     allocation_gt: np.ndarray,
#     source_method: str,
# ) -> List[Dict[str, Any]]:

#     P, W = allocation_gt.shape
#     rows = []

#     load_p = allocation_gt.sum(axis=1)
#     load_d = allocation_gt.sum(axis=0)

#     for p in range(P):
#         gap = int(need[p] - load_p[p])

#         for d in range(W):
#             assigned = int(allocation_gt[p, d])

#             rows.append({
#                 "instance_id": instance_id,
#                 "source_method": source_method,
#                 "period": p,
#                 "driver": d,
#                 "need": int(need[p]),
#                 "local_load": int(load_p[p]),
#                 "driver_load": int(load_d[d]),
#                 "demand_gap": gap,
#                 "label": assigned,
#             })

#     return rows

def build_f1_rows(need, workers_schedule, matrix_allocation):

    mat, ws = _ensure_matrix_and_schedule(
        need, workers_schedule, matrix_allocation
    )

    rows = []
    num_periods, num_workers = mat.shape

    for d in range(num_periods):
        for w in range(num_workers):
            rows.append({
                "period": d,
                "worker": w,
                "need": int(need[d]),
                "allocated_workers": ws[d],
                "worker_total_load": int(mat[:, w].sum()),
                "demand_gap": int(need[d] - ws[d]),
                "label": int(mat[d, w]),  # 0/1
            })

    return rows



# ============================================================
#  BUILD f₂ — Neighborhood-level dataset
# ============================================================

def generate_random_neighborhoods(
    num_periods: int,
    n: int = 5,
    min_size: int = 12,
    max_size: int = 48
):
    neighs = []

    for _ in range(n):
        size = np.random.randint(min_size, max_size)
        end_limit = max(1, num_periods - size)
        start = np.random.randint(0, end_limit)
        neighs.append(list(range(start, start + size)))

    return neighs


# def build_f2_rows(
#     instance_id: int,
#     need: List[int],
#     allocation_gt: np.ndarray,
#     total_workers_greedy: int,
#     total_workers_gt: int,
#     source_method: str,
# ) -> List[Dict[str, Any]]:

#     rows = []
#     P = len(need)

#     neighborhoods = generate_random_neighborhoods(P)

#     for neigh_id, periods in enumerate(neighborhoods):
#         local_load = np.array([allocation_gt[p].sum() for p in periods])
#         uncovered = int(sum(max(0, need[p] - local_load[i]) for i, p in enumerate(periods)))

#         row = {
#             "instance_id": instance_id,
#             "source_method": source_method,
#             "neighborhood_id": neigh_id,
#             "num_periods": len(periods),
#             "uncovered_need": uncovered,
#             "avg_load": float(local_load.mean()),
#             "load_variance": float(local_load.var()),
#             "heur_total_workers": int(total_workers_greedy),
#             "gt_total_workers": int(total_workers_gt),
#             "improvement": int(max(0, total_workers_greedy - total_workers_gt)),
#             "label": int(max(0, total_workers_greedy - total_workers_gt)),
#         }
#         rows.append(row)

#     return rows

def build_f2_rows(need, workers_schedule, matrix_allocation, kpis):

    mat, ws = _ensure_matrix_and_schedule(
        need, workers_schedule, matrix_allocation
    )

    uncovered = np.maximum(np.array(need) - np.array(ws), 0)

    row = {
        "num_periods": len(need),
        "total_demand": int(np.sum(need)),
        "total_allocated": int(np.sum(ws)),
        "uncovered_demand": int(np.sum(uncovered)),
        "avg_load": float(np.mean(ws)),
        "load_variance": float(np.var(ws)),
        "global_coverage": kpis["global_coverage"],
        "worker_efficiency": kpis["worker_efficiency"],
        "operational_risk": kpis["operational_risk"],
        "temporal_stability": kpis["temporal_stability"],
        # label pode ser score, delta ou binário
        "label": 1,  # por agora: solução válida
    }

    return [row]


# ============================================================
#  SAVE CSV
# ============================================================

def write_opt(df: pd.DataFrame, path: str):
    if df.empty:
        return

    exists = os.path.exists(path) and os.path.getsize(path) > 10
    df.to_csv(path, mode="a" if exists else "w", index=False, header=not exists)


def save_datasets(f1_rows, f2_rows):
    write_opt(pd.DataFrame(f1_rows), F1_DATASET_PATH)
    write_opt(pd.DataFrame(f2_rows), F2_DATASET_PATH)
    return F1_DATASET_PATH, F2_DATASET_PATH
