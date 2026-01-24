from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from ml.ml_guidance import neighborhood_scorer


def select_best_neighborhood(
    current_solution,
    candidate_neighborhoods,
    need,
    best_total_active_drivers,
    best_total_assigned_slots=None,
    neighborhood_scorer_fn=None,
):
    scored = []

    for periods_to_free in candidate_neighborhoods:
        if neighborhood_scorer_fn is not None:
            score = call_neighborhood_scorer(
            neighborhood_scorer_fn=neighborhood_scorer_fn,
            current_solution=current_solution,
            periods_to_free=periods_to_free,
            need=need,
            best_total_active_drivers=best_total_active_drivers,
            best_total_assigned_slots=best_total_assigned_slots,
            )
        else:
            # fallback heurístico simples
            score = estimate_uncovered_need(
                current_solution,
                periods_to_free,
                need
            )            
        
        scored.append((score, periods_to_free))

    _, best = max(scored, key=lambda x: x[0])
    return best

# ============================================================
# Universal scorer caller (compat + resiliente)
# ============================================================

def call_neighborhood_scorer(
    neighborhood_scorer_fn: Optional[Callable[..., Any]],
    current_solution: np.ndarray,
    periods_to_free: List[int],
    need: List[int],
    best_total_active_drivers: int,
    best_total_assigned_slots: Optional[int] = None,    
    
) -> Optional[float]:
    """
    Tenta chamar um scorer de vizinhança (legacy ou novo), sem quebrar o LNS.
    """
    if neighborhood_scorer_fn is None:
        return None

    # try:
    #     sig = inspect.signature(neighborhood_scorer_fn)
    #     params = sig.parameters

    #     # Nova API (ml_guidance.neighborhood_scorer)
    #     if {"current_solution", "periods_to_free", "need", "best_total_workers"}.issubset(params):
    #         return float(neighborhood_scorer_fn(
    #             current_solution=current_solution,
    #             periods_to_free=periods_to_free,
    #             need=need,
    #             best_total_active_drivers=best_total_active_drivers,
    #             best_total_assigned_slots=best_total_assigned_slots,
    #         ))

    #     # Legacy: (allocation_matrix, neighborhood_dict)
    #     if {"allocation_matrix", "neighborhood"}.issubset(params):
    #         neighborhood = {"periods": periods_to_free}
    #         return float(neighborhood_scorer_fn(
    #             allocation_matrix=current_solution,
    #             neighborhood=neighborhood,
    #         ))

    #     # fallback posicional
    #     return float(neighborhood_scorer_fn(current_solution, periods_to_free, need, best_total_workers))

    # except Exception:
    #     return None
    
    try:
        sig = inspect.signature(neighborhood_scorer_fn)
        params = sig.parameters

        # --- NOVA API (preferencial, correta) ---
        if {
            "current_solution",
            "periods_to_free",
            "need",
            "best_total_active_drivers",
        }.issubset(params):
            return float(neighborhood_scorer_fn(
                current_solution=current_solution,
                periods_to_free=periods_to_free,
                need=need,
                best_total_active_drivers=best_total_active_drivers,
                best_total_assigned_slots=best_total_assigned_slots,
            ))

        # --- API INTERMEDIÁRIA (usa drivers como métrica principal) ---
        if {
            "current_solution",
            "periods_to_free",
            "need",
            "best_total_workers",
        }.issubset(params):
            return float(neighborhood_scorer_fn(
                current_solution=current_solution,
                periods_to_free=periods_to_free,
                need=need,
                best_total_workers=best_total_active_drivers,  # 🔥 mapeamento correto
            ))

        # --- LEGACY (inalterado) ---
        if {"allocation_matrix", "neighborhood"}.issubset(params):
            neighborhood = {"periods": periods_to_free}
            return float(neighborhood_scorer_fn(
                allocation_matrix=current_solution,
                neighborhood=neighborhood,
            ))

        # --- fallback posicional ---
        return float(
            neighborhood_scorer_fn(
                current_solution,
                periods_to_free,
                need,
                best_total_active_drivers
            )
        )

    except Exception:
        return None
    
# ============================================================
# Auxiliares LNS
# ============================================================

def build_fixed_assignments_from_solution(
    current_solution: np.ndarray,
    periods_to_free: List[int]
) -> List[Tuple[int, int, int]]:
    """
    Cria lista (period, driver, val) para todas as decisões que NÃO serão liberadas.
    Ou seja: fixa todas as entradas fora da vizinhança.
    """
    fixed: List[Tuple[int, int, int]] = []
    num_periods, num_workers = current_solution.shape

    free_set = set(int(p) for p in periods_to_free)

    for p in range(num_periods):
        if p in free_set:
            continue
        for w in range(num_workers):
            fixed.append((p, w, int(current_solution[p, w])))

    return fixed

def estimate_uncovered_need(solution: np.ndarray, periods: List[int], need: List[int]) -> int:
    """
    Soma de max(0, demanda - cobertura) dentro da vizinhança.
    """
    unc = 0
    for p in periods:
        if 0 <= p < solution.shape[0]:
            dem = int(need[p]) if p < len(need) else 0
            cov = int(solution[p, :].sum())
            unc += max(0, dem - cov)
    return int(unc)

def generate_candidate_neighborhoods(
    num_periods: int,
    periods_per_day: int = 96,
    k: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> List[List[int]]:
    """
    Gera k vizinhanças candidatas (por dia), escolhendo dias aleatórios.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_days = max(1, num_periods // periods_per_day)
    k = max(1, min(k, num_days))

    days = rng.choice(np.arange(num_days), size=k, replace=False).tolist()
    neighborhoods: List[List[int]] = []
    for day in days:
        start = day * periods_per_day
        end = min(num_periods, start + periods_per_day)
        neighborhoods.append(list(range(start, end)))
    return neighborhoods

# ============================================================
# LNS principal
# ============================================================

def run_lns(
    initial_solution: np.ndarray,
    need: List[int],
    variable_type: str,
    constraints_coefficients: Any,
    selected_restrictions: Dict[str, bool],
    solver_param_type: str,
    limit_workers: int,
    limit_iteration: int,
    limit_level_relaxation: int,
    max_demands_per_driver: int,
    tolerance_demands: float,
    penalty: float,
    max_lns_iterations: int,
    solve_fn: Callable[..., Any],
    neighborhood_scorer_fn: Optional[Callable[..., Any]] = None,
    candidate_k: int = 6,
    periods_per_day: int = 96,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executa Large Neighborhood Search (LNS) guiado por heurística + ML (f₂).

    Retorna
    -------
    best_solution : np.ndarray
    info : Dict com histórico, best_total_workers e last_solver_logs
    """
    rng = np.random.default_rng(random_seed)

    num_periods = len(need)
    # best_solution = initial_solution.copy()
    
    
    best_solution = np.asarray(initial_solution)

    # Garante shape 2D: (num_periods, num_workers)
    if best_solution.ndim == 1:
        # Tentativa 1: se for coverage por slot (len == num_periods),
        # não dá para reconstruir matriz sem num_workers -> erro claro.
        raise ValueError(
            f"LNS espera initial_solution 2D (periods x drivers), "
            f"mas recebeu 1D shape={best_solution.shape}. "
            f"Passe matrix_allocation (2D) como initial_solution."
        )

    if best_solution.ndim != 2:
        raise ValueError(
            f"LNS espera initial_solution 2D, mas recebeu ndim={best_solution.ndim}, "
            f"shape={best_solution.shape}"
        )

    best_solution = best_solution.copy()    
    
    best_total_assigned_slots = int(best_solution.sum())
    best_total_active_drivers = int(np.sum(np.any(best_solution > 0, axis=0)))
    
    # best_total_workers = int(best_solution.sum())

    history: List[Dict[str, Any]] = []
    last_solver_logs = {"stdout": "", "stderr": ""}

    for it in range(int(max_lns_iterations)):

        # 1) Gerar vizinhanças candidatas
        candidate_neighborhoods = generate_candidate_neighborhoods(
            num_periods=num_periods,
            periods_per_day=periods_per_day,
            k=candidate_k,
            rng=rng,
        )
        
        
        periods_to_free = select_best_neighborhood(
            current_solution=best_solution,
            candidate_neighborhoods=candidate_neighborhoods,
            need=need,
            best_total_active_drivers=best_total_active_drivers,
            best_total_assigned_slots=best_total_assigned_slots,
            neighborhood_scorer_fn=neighborhood_scorer_fn,
        )

        # 2) Scorar vizinhanças (ML se existir; senão heurística)
        scored: List[Tuple[float, List[int]]] = []
        
        for periods_to_free in candidate_neighborhoods:
            ml_score = call_neighborhood_scorer(
                neighborhood_scorer_fn,
                current_solution=best_solution,
                periods_to_free=periods_to_free,
                need=need,
                best_total_active_drivers=best_total_active_drivers,
                best_total_assigned_slots=best_total_assigned_slots,
            )
            
            base_score = float(estimate_uncovered_need(best_solution, periods_to_free, need))
            final_score = float(ml_score) if ml_score is not None else base_score
            scored.append((final_score, periods_to_free))

        # se por algum motivo não houver candidatos
        if not scored:
            break

        # 3) Escolher melhor vizinhança (maior score)
        _, periods_to_free = max(scored, key=lambda x: x[0])

        # 4) Fixar tudo fora da vizinhança
        fixed_assignments = build_fixed_assignments_from_solution(
            current_solution=best_solution,
            periods_to_free=periods_to_free
        )

        # 5) Resolver subproblema (MILP com variáveis fixadas)
        # (
        #     solver,
        #     status,
        #     total_active_drivers,      # ← decisão
        #     total_assigned_slots,      # ← esforço
        #     workers_schedule,
        #     new_constraints,
        #     initial_density,
        #     final_density,
        #     statistics_result,
        #     msg,
        #     iterations_data,
        #     matrix_allocation,
        #     solver_logs
        # ) = solve_fn(
        #     solver_param_type,
        #     need,
        #     variable_type,
        #     constraints_coefficients,
        #     selected_restrictions,
        #     swap_rows=None,
        #     multiply_row=None,
        #     add_multiple_rows=None,
        #     densidade_aceitavel=None,
        #     limit_workers=limit_workers,
        #     limit_iteration=limit_iteration,
        #     limit_level_relaxation=limit_level_relaxation,
        #     cap_tasks_per_driver_per_slot=max_demands_per_driver,
        #     tolerance_demands=tolerance_demands,
        #     penalty=penalty,
        #     initial_allocation=None,
        #     fixed_assignments=fixed_assignments,
        #     mode="LNS",
        # )

        result = solve_fn(
            solver_param_type,
            need,
            variable_type,
            constraints_coefficients,
            selected_restrictions,
            swap_rows=None,
            multiply_row=None,
            add_multiple_rows=None,
            densidade_aceitavel=None,
            limit_workers=limit_workers,
            limit_iteration=limit_iteration,
            limit_level_relaxation=limit_level_relaxation,
            cap_tasks_per_driver_per_slot=max_demands_per_driver,
            tolerance_demands=tolerance_demands,
            penalty=penalty,
            initial_allocation=None,
            fixed_assignments=fixed_assignments,
            mode="LNS",
        )

        # ---- unpack defensivo (compatível com solver evoluído) ----
        solver               = result[0]
        status               = result[1]
        total_active_drivers = result[2]
        total_assigned_slots = result[3]
        workers_schedule     = result[4]
        new_constraints      = result[5]
        initial_density      = result[6]
        final_density        = result[7]
        statistics_result    = result[8]
        msg                  = result[9]
        iterations_data      = result[10]
        matrix_allocation    = result[11]
        solver_logs          = result[12]

        last_solver_logs = solver_logs if isinstance(solver_logs, dict) else last_solver_logs

        # 6) Avaliar melhoria
        if matrix_allocation is None:
            history.append({
                "iteration": it,
                "total_active_drivers": int(total_active_drivers) if total_active_drivers is not None else None,
                "total_assigned_slots": int(total_assigned_slots) if total_assigned_slots is not None else None,
                "improved": False,
                "status": status
            })
            continue

        # new_solution = np.asarray(matrix_allocation, dtype=int)
        
        # --------------------------------------------------
        # VALIDAÇÃO CRÍTICA: LNS só aceita solução 2D
        # --------------------------------------------------
        if matrix_allocation is None:
            history.append({
                "iteration": it,
                "status": status,
                "improved": False,
                "reason": "matrix_allocation is None"
            })
            continue

        new_solution = np.asarray(matrix_allocation)

        if new_solution.ndim != 2:
            history.append({
                "iteration": it,
                "status": status,
                "improved": False,
                "reason": f"invalid solution shape {new_solution.shape}"
            })
            continue

        new_solution = new_solution.astype(int)
        
        
        

        new_total_assigned_slots = int(new_solution.sum())
        new_total_active_drivers = int(np.sum(np.any(new_solution > 0, axis=0)))

        # driver-first, depois slots como desempate (opcional mas recomendado)
        improved = (
            (new_total_active_drivers < best_total_active_drivers) or
            (
                new_total_active_drivers == best_total_active_drivers and
                new_total_assigned_slots < best_total_assigned_slots
            )
        )

        # new_solution = np.asarray(matrix_allocation, dtype=int)
        
        # new_total_workers = int(new_solution.sum())
        # improved = new_total_workers < best_total_workers

        # new_total_active_drivers = int(
        #     np.sum(np.any(new_solution > 0, axis=0))
        # )

        # improved = new_total_active_drivers < best_total_workers

        history.append({
            "iteration": it,
            "total_active_drivers": new_total_active_drivers,
            "total_assigned_slots": new_total_assigned_slots,
            "improved": improved,
            "status": status
        })

        if improved:
            best_solution = new_solution
            best_total_active_drivers = new_total_active_drivers
            best_total_assigned_slots = new_total_assigned_slots

    info = {
        "total_active_drivers": int(best_total_active_drivers),
        "total_assigned_slots": int(best_total_assigned_slots),
        "history": history,
        "last_solver_logs": last_solver_logs
    }
    return best_solution, info
