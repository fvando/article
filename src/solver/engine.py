import io
import contextlib
import json
import re
import os
import time
from typing import Any, List, Optional, Tuple, Dict
import numpy as np
from ortools.linear_solver import pywraplp
from src.core.i18n import t

from src.solver.heuristic import greedy_initial_allocation
from src.solver.lns import run_lns
from src.ml.ml_guidance import assignment_scorer, neighborhood_scorer

# ==============================================================================
#  PERSISTÊNCIA E HELPERS (ESTILO V9.6)
# ==============================================================================

def load_data(filename):
    try:
        if not os.path.exists(filename):
            return np.array([])
        with open(filename, 'r') as f:
            data = json.load(f)
            return np.array(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return np.array([])

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data.tolist() if isinstance(data, np.ndarray) else data, f)

def calculate_density(matrix):
    if matrix is None or len(matrix) == 0:
        return 0
    if isinstance(matrix, np.ndarray):
        total = matrix.size
        non_zero = np.count_nonzero(matrix)
    else:
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        total = rows * cols
        non_zero = sum(1 for r in matrix for c in r if c != 0)
    return non_zero / total if total > 0 else 0

def tipo_modelo(solver):
    return "Exact (MILP)"

def get_solver_status_description(status):
    status_mapping = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }
    return status_mapping.get(status, "UNKNOWN_STATUS")

def parse_statistics_list(statistics_result):
    out = {}
    for line in statistics_result:
        if not isinstance(line, str): continue
        s = line.strip()
        if not s: continue
        
        if ":" in s: k, v = s.split(":", 1)
        elif "=" in s: k, v = s.split("=", 1)
        else:
            parts = s.split(None, 1)
            if len(parts) != 2: continue
            k, v = parts[0], parts[1]

        key = k.strip().lower()
        val = v.strip()
        
        # Mapeamento chave -> campo normalizado
        key = key.replace("model state", "status_text")\
                 .replace("total active drivers", "total_active_drivers")\
                 .replace("total assigned slots", "total_assigned_slots")\
                 .replace("total resolution time", "solve_time_ms")\
                 .replace("total number of iterations", "iterations")\
                 .replace("number of restrictions", "num_constraints")\
                 .replace("number of variables", "num_vars")\
                 .replace("objective value", "objective_value")\
                 .replace("best bound", "best_bound")\
                 .replace("gap (%)", "gap_percent")\
                 .replace("solve time (s)", "solve_time_s")\
                 .replace("stopped by time/gap", "stopped_by_limit")\
                 .replace("density check", "density_check")

        # Parse valores
        if key in ("total_active_drivers", "total_assigned_slots", "iterations", "num_constraints", "num_vars"):
            m = re.search(r"-?\d+", val)
            out[key] = int(m.group()) if m else None
        elif key in ("objective_value", "best_bound", "gap_percent", "solve_time_s", "solve_time_ms"):
            m = re.search(r"-?\d+(\.\d+)?", val.replace(",", "."))
            out[key] = float(m.group()) if m else None
        elif key == "stopped_by_limit":
            out[key] = val.lower() in ("true", "1", "yes", "sim")
        elif key == "density_check":
             m_final = re.search(r"final\s*=\s*([0-9.]+)", val)
             m_thr = re.search(r"threshold\s*=\s*([0-9.]+)", val)
             out["density"] = float(m_final.group(1)) if m_final else None
             out["density_threshold"] = float(m_thr.group(1)) if m_thr else None
             out[key] = val
        else:
            out[key] = val

    # Status Enum
    stxt = (out.get("status_text") or "").upper()
    if "OPTIMAL" in stxt: out["status"] = pywraplp.Solver.OPTIMAL
    elif "FEASIBLE" in stxt: out["status"] = pywraplp.Solver.FEASIBLE
    elif "INFEASIBLE" in stxt: out["status"] = pywraplp.Solver.INFEASIBLE
    else: out["status"] = None

    if out.get("gap_percent") is not None:
        out["gap"] = out["gap_percent"] / 100.0
    else:
        out["gap"] = None
        
    return out

def interpret_solver_result(result):
    # aceita dict OU list[str]
    if isinstance(result, list):
        result = parse_statistics_list(result)
    if not isinstance(result, dict):
        return {"model_state": t("eng_err_interp")}

    interpretation = {}

    status = result.get("status")
    gap = result.get("gap")

    if status == pywraplp.Solver.OPTIMAL and (gap is not None and gap <= 0.01):
        state_msg = t("eng_opt_proven")
    elif status == pywraplp.Solver.OPTIMAL:
        state_msg = t("eng_opt_likely").format(gap*100 if gap else 0)
    elif status == pywraplp.Solver.FEASIBLE:
        state_msg = t("eng_feasible")
    elif status == pywraplp.Solver.INFEASIBLE:
        state_msg = t("eng_infeasible")
    elif status == "HEURISTIC":
        state_msg = t("eng_heuristic_state")
    else:
        state_msg = t("eng_unknown")

    interpretation["model_state"] = state_msg

    # Métricas operacionais
    assigned = result.get("total_assigned_slots") or 0
    drivers = result.get("total_active_drivers") or 0
    total_hours = assigned * 0.25
    avg_hours = total_hours / drivers if drivers > 0 else 0.0

    interpretation["operational_metrics"] = {
        "drivers_used": drivers,
        "assigned_slots": assigned,
        "total_hours": round(total_hours, 2),
        "avg_hours_per_driver": round(avg_hours, 2),
    }

    # Qualidade
    quality = []
    if gap is None:
        quality.append(t("eng_gap_none"))
    elif gap > 0.05:
        quality.append(t("eng_gap_high").format(gap*100))
    elif gap > 0.01:
        quality.append(t("eng_gap_mod").format(gap*100))
    else:
        quality.append(t("eng_gap_low").format(gap*100))

    interpretation["solution_quality"] = quality

    # --------------------------------------------------
    # Estrutura do modelo (explicação conceitual)
    # --------------------------------------------------
    model_structure = []
    model_structure.append(t("eng_model_desc"))
    if result.get("num_constraints") is not None:
        model_structure.append(
            t("eng_constr_active").format(result['num_constraints'])
        )
    interpretation["model_structure"] = model_structure
    
    # --------------------------------------------------
    # Comportamento do Solver
    # --------------------------------------------------
    solver_behavior = []
    solver_behavior.append(t("eng_bnb_desc"))
    if status == pywraplp.Solver.OPTIMAL:
        solver_behavior.append(t("eng_opt_reached"))
    
    interpretation["solver_behavior"] = solver_behavior

    return interpretation

def format_lp_output(num_vars, num_restricoes, rhs_values):
    output = f"NumVars:{num_vars}\n\nNumRestrictions:{num_restricoes}\n\nNumrhs_values:{len(rhs_values)}\n\nMODEL:\n\n [_1] MIN= " + " + ".join(f"X_{i+1}" for i in range(num_vars)) + ";\n\n"
    restricao = " + ".join(f"X_{i+1}" for i in range(num_vars))
    for j in range(num_vars):
        rhs_value = rhs_values[j] if j < len(rhs_values) else 0
        output += f" [_{j+2}] {restricao} >= {rhs_value};\n\n"
    output += "\n" + " ".join(f"@GIN(X_{i+1});" for i in range(num_vars)) + "\nEND\n"
    return output

def rebuild_allocation_from_schedule(workers_schedule, need, limit_workers):
    num_periods = len(workers_schedule)
    allocation = np.zeros((num_periods, limit_workers), dtype=int)
    for p, w in enumerate(workers_schedule):
        w_val = min(int(round(float(w))), limit_workers)
        allocation[p, :w_val] = 1
    return allocation

def normalize_solver_outputs(demanda, workers_schedule, matrix_allocation):
    if matrix_allocation is not None:
        try:
            matrix_allocation = np.asarray(matrix_allocation, dtype=int)
            if matrix_allocation.ndim != 2: matrix_allocation = None
        except: matrix_allocation = None

    if matrix_allocation is None and workers_schedule is not None:
        try: matrix_allocation = np.array(workers_schedule, dtype=int).reshape(-1, 1)
        except: matrix_allocation = None
    
    ws = None
    if matrix_allocation is not None:
        ws = list(np.sum(matrix_allocation, axis=1).astype(int))
    elif isinstance(workers_schedule, (list, np.ndarray)):
        ws = [int(x) for x in workers_schedule]

    if ws is not None and len(ws) != len(demanda): ws = None
    return ws, matrix_allocation

# ==============================================================================
#  CORE: SOLVE_SHIFT_SCHEDULE (100% PROXY V9.6)
# ==============================================================================

def solve_shift_schedule(
    solver_param_type, need, variable_type, constraints_coefficients, selected_restrictions, 
    swap_rows=None, multiply_row=None, add_multiple_rows=None, densidade_aceitavel=None, 
    limit_workers=0, limit_iteration=0, limit_level_relaxation=0, 
    cap_tasks_per_driver_per_slot=1, tolerance_demands=0.01, penalty = 0.01,
    initial_allocation=None, fixed_assignments=None, radio_selection_object=None,
    mode="Exact", enable_symmetry_breaking=False, time_limit_seconds=300,
    enable_assignment_maximization=True
):
    if time_limit_seconds is None:
        time_limit_seconds = 300
    
    # Normalização de selected_restrictions para dicionário
    if isinstance(selected_restrictions, list):
        selected_restrictions = {r: True for r in selected_restrictions}
    elif selected_restrictions is None:
        selected_restrictions = {}
        
    constraints = []
    num_periods = len(need)
    Y = {}
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with contextlib.redirect_stdout(stdout_buffer):
        with contextlib.redirect_stderr(stderr_buffer):    
            solver = pywraplp.Solver.CreateSolver(solver_param_type)
            solver.EnableOutput()
            
            # FIX SCIP PARAMETERS - Robust parsing
            if "SCIP" in str(solver_param_type).upper():
                # SCIP parameters must be 'name = value' or 'name=value'
                # We use the most standard format
                param_str = f"limits/time = {time_limit_seconds}\nlimits/gap = 0.05\n"
                solver.SetSolverSpecificParametersAsString(param_str)
            else:
                solver.SetTimeLimit(int(time_limit_seconds * 1000))
            
            print(f"DEBUG ENGINE: Created solver {solver_param_type} ({tipo_modelo(solver)})")
            
            # DEBUG LOGGING
            print(f"\n=== SOLVER DEBUG ===")
            print(f"num_periods: {num_periods}")
            print(f"limit_workers: {limit_workers}")
            print(f"Total demand (sum): {float(sum(need))}")
            print(f"Max demand in any slot: {float(max(need))}")
            print(f"cap_tasks_per_driver_per_slot: {cap_tasks_per_driver_per_slot}")
            print(f"selected_restrictions: {list(selected_restrictions.keys())}")
            print(f"variable_type: {variable_type}")
            
            # CRITICAL FIX: cobertura_necessidade MUST always be True
            # This is a fundamental constraint that balances demand and supply
            selected_restrictions['cobertura_necessidade'] = True
            print(f"⚠️  FORCED cobertura_necessidade = True (fundamental constraint)")
            
            Z = {t: solver.BoolVar(f"Z_{t}") for t in range(limit_workers)}
            U = {d: solver.NumVar(0, need[d], f"U[{d}]") for d in range(num_periods)}
            X = {}
            for d in range(num_periods):
                for t in range(limit_workers):
                    X[d, t] = solver.IntVar(0, cap_tasks_per_driver_per_slot, f"X[{d},{t}]")
            for d in range(num_periods):
                for t in range(limit_workers):
                    Y[d, t] = solver.NumVar(0, 1, f'Y[{d},{t}]') if variable_type == "Continuous" else solver.BoolVar(f'Y[{d},{t}]')

            # WARM START (HINTS)
            if initial_allocation is not None and mode == "Exact":
                if initial_allocation.shape == (num_periods, limit_workers):
                    print(f"DEBUG: Setting solver hints for active tasks (X > 0)...")
                    hint_vars = []
                    hint_vals = []
                    
                    for d in range(num_periods):
                        for t in range(limit_workers):
                            val = initial_allocation[d, t]
                            if val > 0:
                                # Only hint assigned tasks. 
                                # Hinting 0s for Y can conflict with SHIFT_MIN constraints in some presolvers.
                                hint_vars.append(X[d, t])
                                hint_vals.append(float(val))
                    
                    if hasattr(solver, 'SetHint') and hint_vars:
                        try:
                            # Robust conversion to lists for OR-Tools SWIG wrapper
                            v_list = list(hint_vars)
                            val_list = [float(v) for v in hint_vals]
                            solver.SetHint(v_list, val_list)
                            print(f"DEBUG: Successfully set {len(v_list)} variable hints.")
                        except Exception as e:
                            print(f"⚠️  WARNING: Hint logic ignored: {e}")

            for d in range(num_periods):
                for t in range(limit_workers):
                    solver.Add(X[d, t] <= float(cap_tasks_per_driver_per_slot) * Y[d, t])
                    solver.Add(Y[d, t] <= Z[t])
            for t in range(limit_workers):
                solver.Add(solver.Sum([Y[d, t] for d in range(num_periods)]) <= float(num_periods) * Z[t])

            L = {t: solver.NumVar(0, float(cap_tasks_per_driver_per_slot * num_periods), f"L[{t}]") for t in range(limit_workers)}
            for t in range(limit_workers):
                solver.Add(L[t] == solver.Sum([X[d, t] for d in range(num_periods)]))
                solver.Add(L[t] <= float(cap_tasks_per_driver_per_slot * num_periods) * Z[t]) 
            Lmax = solver.NumVar(0, cap_tasks_per_driver_per_slot * num_periods, "Lmax")
            for t in range(limit_workers): solver.Add(L[t] <= Lmax)

            S, E = {}, {}
            SHIFT_MIN = min(36, num_periods)  # Dynamic: avoid infeasibility on short horizons
            SHIFT_MAX = min(52, num_periods)
            for t in range(limit_workers):
                for d in range(num_periods):
                    S[d, t], E[d, t] = solver.BoolVar(f"S[{d},{t}]"), solver.BoolVar(f"E[{d},{t}]")
                solver.Add(solver.Sum([S[d, t] for d in range(num_periods)]) == Z[t])
                solver.Add(solver.Sum([E[d, t] for d in range(num_periods)]) == Z[t])
                solver.Add(solver.Sum([Y[d, t] for d in range(num_periods)]) >= float(SHIFT_MIN) * Z[t])
                solver.Add(solver.Sum([Y[d, t] for d in range(num_periods)]) <= float(SHIFT_MAX) * Z[t])
                solver.Add(E[0, t] == 0)
                solver.Add(Y[0, t] == S[0, t])
                for d in range(1, num_periods): solver.Add(Y[d, t] - Y[d - 1, t] == S[d, t] - E[d, t])

            Over = {}
            Dev = {}
            SOFT_CAP = 32 * cap_tasks_per_driver_per_slot
            for t in range(limit_workers):
                Over[t] = solver.NumVar(0, cap_tasks_per_driver_per_slot * num_periods, f"Over[{t}]")
                Dev[t] = solver.NumVar(0, cap_tasks_per_driver_per_slot * num_periods, f"Dev[{t}]")
                
                solver.Add(Over[t] >= L[t] - SOFT_CAP)
                solver.Add(Over[t] >= 0)
                solver.Add(Over[t] <= cap_tasks_per_driver_per_slot * num_periods * Z[t])
                
                solver.Add(Dev[t] >= SOFT_CAP - L[t])
                solver.Add(Dev[t] >= 0)
                solver.Add(Dev[t] <= cap_tasks_per_driver_per_slot * num_periods * Z[t])

            # ==========================================================================
            # PHASE 1: Minimize Unmet Demand (Coverage Priority)
            # ==========================================================================
            objective = solver.Objective()
            objective.SetMinimization()
            
            for d in range(num_periods):
                # Primary goal: Minimize total U[d]
                # We add a tiny gradient to prefer covering early slots if tied
                objective.SetCoefficient(U[d], float(1.0 + 0.0001 * (d / num_periods)))
            
            print(f"DEBUG ENGINE: PHASE 1 Setup (Minimize Unmet Demand)")

            # Coverage Constraint
            print(f"DEBUG ENGINE: Sum of demand (need): {float(sum(need))}")
            for d in range(num_periods): 
                solver.Add(solver.Sum([X[d, t] for t in range(limit_workers)]) + U[d] == float(need[d]))
            
            # Pausa de 45 minutos (4h30 de condução + 45min de pausa)
            if selected_restrictions.get("pausa_45_minutos", False):
                max_continuous_work = 18  # 4h30 = 18 slots
                pause_duration = 3  # 45 min = 3 slots
                window = max_continuous_work + pause_duration  # 21
                for t in range(limit_workers):
                    for start in range(num_periods - window + 1):
                        solver.Add(solver.Sum([X[start + p, t] for p in range(window)]) <= float(max_continuous_work))
            
            # Limite diário de condução (9 horas por dia)
            if selected_restrictions.get("limite_diario", False):
                max_daily_driving = 36  # 9 horas (36 períodos de 15 minutos)
                periods_per_day = 96  # 24 horas
                for day in range(num_periods // periods_per_day):
                    day_start = day * periods_per_day
                    day_end = (day + 1) * periods_per_day
                    for t in range(limit_workers):
                        solver.Add(solver.Sum([X[d, t] for d in range(day_start, day_end)]) <= float(max_daily_driving))
            
            # Repouso diário mínimo (11h descanso)
            if selected_restrictions.get("repouso_diario_minimo", False):
                periods_per_day = 96
                max_work_normal = 52   # 96 - 44 (11h repouso)
                max_work_reduced = 60  # 96 - 36 (9h repouso)
                reduced_rest_days = 3  # no máximo 3 repousos reduzidos
                
                for t in range(limit_workers):
                    reduced_rest_vars = []
                    for day in range(num_periods // periods_per_day):
                        reduced_rest = solver.BoolVar(f"reduced_rest_d{day}_t{t}")
                        reduced_rest_vars.append(reduced_rest)
                        
                        day_start = day * periods_per_day
                        day_end = day_start + periods_per_day
                        daily_work = solver.Sum([X[d, t] for d in range(day_start, day_end)])
                        
                        solver.Add(daily_work <= float(max_work_reduced) * reduced_rest + float(max_work_normal) * (1 - reduced_rest))
                    
                    # no máximo 3 dias com repouso diário reduzido
                    solver.Add(solver.Sum(reduced_rest_vars) <= float(reduced_rest_days))
            
            # Limite semanal (56h por semana)
            if selected_restrictions.get("limite_semanal", False):
                periods_per_day = 96
                periods_per_week = 96 * 7
                max_weekly_driving = 224  # 56h = 224 períodos
                
                for week in range(num_periods // periods_per_week):
                    week_start = week * periods_per_week
                    week_end = (week + 1) * periods_per_week
                    for t in range(limit_workers):
                        solver.Add(solver.Sum(X[d, t] for d in range(week_start, week_end)) <= max_weekly_driving)
            
            # Repouso diário reduzido (24h descanso semanal)
            if selected_restrictions.get("repouso_diario_reduzido", False):
                periods_per_day = 96
                periods_per_week = periods_per_day * 7
                min_weekly_rest = 96  # 24h = 96 períodos
                max_reduced_weeks = 1  # no máximo 1 repouso semanal reduzido (em 2 semanas)
                
                for t in range(limit_workers):
                    reduced_week_vars = []
                    for week in range(num_periods // periods_per_week):
                        reduced_week = solver.BoolVar(f"reduced_week_rest_w{week}_t{t}")
                        reduced_week_vars.append(reduced_week)
                        
                        week_start = week * periods_per_week
                        week_end = week_start + periods_per_week
                        weekly_work = solver.Sum(Y[d, t] for d in range(week_start, week_end))
                        
                        # trabalho máximo permitido na semana
                        solver.Add(weekly_work <= periods_per_week - min_weekly_rest * (1 - reduced_week))
                    
                    # no máximo 1 semana com repouso semanal reduzido
                    solver.Add(solver.Sum(reduced_week_vars) <= max_reduced_weeks)
            
            # Descanso após trabalho (mínimo 24h/semana)
            if selected_restrictions.get("descanso_apos_trabalho", False):
                periods_per_day = 96
                periods_per_week = periods_per_day * 7
                min_weekly_rest = 96  # 24h de repouso semanal reduzido (mínimo legal)
                
                for t in range(limit_workers):
                    for week in range(num_periods // periods_per_week):
                        week_start = week * periods_per_week
                        week_end = week_start + periods_per_week
                        weekly_work = solver.Sum(Y[d, t] for d in range(week_start, week_end))
                        
                        # deve existir pelo menos 24h (96 períodos) sem trabalho
                        solver.Add(weekly_work <= periods_per_week - min_weekly_rest)
            
            # Limite quinzenal (90h em 2 semanas)
            if selected_restrictions.get("limite_quinzenal", False):
                periods_per_day = 96
                periods_per_week = periods_per_day * 7
                periods_per_fortnight = periods_per_week * 2
                max_fortnight_driving = 360  # 90h = 360 períodos de 15 min
                
                for fortnight in range(num_periods // periods_per_fortnight):
                    fortnight_start = fortnight * periods_per_fortnight
                    fortnight_end = (fortnight + 1) * periods_per_fortnight
                    for t in range(limit_workers):
                        solver.Add(solver.Sum(Y[d, t] for d in range(fortnight_start, fortnight_end)) <= max_fortnight_driving)
            
            # Repouso quinzenal (24h a cada 2 semanas)
            if selected_restrictions.get("repouso_quinzenal", False):
                periods_per_day = 96
                periods_per_week = periods_per_day * 7
                periods_per_fortnight = periods_per_week * 2
                min_fortnight_rest = 96  # 24h = 96 períodos
                
                for t in range(limit_workers):
                    for fortnight in range(num_periods // periods_per_fortnight):
                        fortnight_start = fortnight * periods_per_fortnight
                        fortnight_end = (fortnight + 1) * periods_per_fortnight
                        fortnight_work = solver.Sum(Y[d, t] for d in range(fortnight_start, fortnight_end))
                        
                        # deve existir pelo menos 24h (96 períodos) sem trabalho a cada 2 semanas
                        solver.Add(fortnight_work <= periods_per_fortnight - min_fortnight_rest)
            
            # Repouso semanal (45h por semana)
            if selected_restrictions.get("repouso_semanal", False):
                periods_per_day = 96
                periods_per_week = periods_per_day * 7
                min_weekly_rest = 180  # 45h = 180 períodos
                
                for t in range(limit_workers):
                    for week in range(num_periods // periods_per_week):
                        week_start = week * periods_per_week
                        week_end = week_start + periods_per_week
                        weekly_work = solver.Sum(Y[d, t] for d in range(week_start, week_end))
                        
                        # deve existir pelo menos 45h (180 períodos) sem trabalho por semana
                        solver.Add(weekly_work <= periods_per_week - min_weekly_rest)

            # ==========================================================================
            # LEXICOGRAPHICAL RESOLUTION - 2 PHASES
            # ==========================================================================
            print(f"DEBUG ENGINE: Phase 1 Solving (Coverage)...")
            
            # Use 40% of time for Phase 1
            if "SCIP" in str(solver_param_type).upper():
                solver.SetSolverSpecificParametersAsString(f"limits/time = {time_limit_seconds * 0.4}\n")
            else:
                solver.SetTimeLimit(int(time_limit_seconds * 0.4 * 1000))

            status = solver.Solve()
            
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                min_unmet = sum(U[d].solution_value() for d in range(num_periods))
                print(f"✅ PHASE 1 COMPLETE. Min Unmet Demand: {min_unmet:.2f}")
                
                # Lock Unmet Demand (lexicographical requirement)
                # Allow 0.01% epsilon or 0.1 slot to avoid numerical infeasibility in Phase 2
                solver.Add(solver.Sum([U[d] for d in range(num_periods)]) <= float(min_unmet + 0.1))
                
                # PHASE 2: Minimize Resource Use
                objective.Clear()
                objective.SetMinimization()
                
                # Resource Priority: Number of active drivers
                for t in range(limit_workers):
                    objective.SetCoefficient(Z[t], float(1.0))
                
                # Tertiary goal: Maximize assignments (to cover optional capacity)
                # This uses a small reward to break ties in driver count
                if enable_assignment_maximization:
                    for d in range(num_periods):
                        for t in range(limit_workers):
                            objective.SetCoefficient(X[d, t], -0.001)

                print(f"DEBUG ENGINE: Phase 2 Solving (Resource Optimization)...")
                
                # Use remaining 60% of time for Phase 2
                if "SCIP" in str(solver_param_type).upper():
                    solver.SetSolverSpecificParametersAsString(f"limits/time = {time_limit_seconds * 0.6}\n")
                else:
                    solver.SetTimeLimit(int(time_limit_seconds * 0.6 * 1000))
                
                status = solver.Solve()
                print(f"✅ PHASE 2 COMPLETE. Final Status: {status}")
            else:
                print(f"⚠️  PHASE 1 FAILED (status={status}). Skipping Phase 2.")

            matrix_allocation = np.zeros((num_periods, limit_workers), dtype=int)
            workers_schedule, tasks_schedule = [0] * num_periods, [0] * num_periods
            driving_hours_per_driver = {t: 0.0 for t in range(limit_workers)}
            total_active, total_slots = 0, 0
            statistics_result = []
            
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                # USE X (TASKS) FOR ALLOCATION AND SLOTS FOR FAIR COMPARISON WITH HEURISTIC
                # matrix_allocation must store the actual integer count to fix coverage reporting in kpis.py
                matrix_allocation = np.array([[int(round(X[d, t].solution_value())) for t in range(limit_workers)] for d in range(num_periods)], dtype=int)
                
                # workers_schedule must represent drivers present (count of drivers per period)
                # If matrix_allocation stores counts, ws[d] = sum(1 if matrix_allocation[d, t] > 0 else 0)
                workers_schedule = [int(sum(1 for val in matrix_allocation[d, :] if val > 0)) for d in range(num_periods)]
                
                # tasks_schedule represents total tasks per period
                tasks_schedule = [float(sum(matrix_allocation[d, :])) for d in range(num_periods)]
                
                total_active = int(sum(1 for t in range(limit_workers) if Z[t].solution_value() > 0.5))
                total_slots = int(round(sum(tasks_schedule)))
                
                print(f"DEBUG FINAL: total_slots={total_slots}, tasks_sum={sum(tasks_schedule)}, unmet_sum={sum(U[d].solution_value() for d in range(num_periods))}")
            else:
                 # FALLBACK: If solver fails (Timeout/Infeasible) return the initial allocation (Heuristic)
                 # This prevents the "No drivers allocated" issue which breaks the UI
                print(f"⚠️ SOLVER FAILED or TIMED OUT (Status={status}). Fallback to initial allocation (Heuristic).")
                if initial_allocation is not None:
                     matrix_allocation = initial_allocation
                     # Recalculate basic stats from fallback
                     workers_schedule = [int(sum(1 for val in matrix_allocation[d, :] if val > 0)) for d in range(num_periods)]
                     tasks_schedule = [float(sum(matrix_allocation[d, :])) for d in range(num_periods)]
                     total_active = int(np.sum(np.any(matrix_allocation > 0, axis=0)))
                     total_slots = int(round(matrix_allocation.sum()))
                     
                     statistics_result.append(f"Model State: FALLBACK (Heuristic)")
                else:
                    print("⚠️ NO FALLBACK AVAILABLE. Returning empty solution.")

                
                for t in range(limit_workers): driving_hours_per_driver[t] = float(sum(X[d, t].solution_value() for d in range(num_periods)) * 0.25)
                
                # ===== VALIDAÇÃO DE RESTRIÇÕES =====
                print(f"\n=== CONSTRAINT VALIDATION ===")
                violations = []
                
                # Validar limite_diario (9h/dia)
                if selected_restrictions.get("limite_diario", False):
                    periods_per_day = 96
                    max_daily = 36
                    for day in range(num_periods // periods_per_day):
                        for t in range(limit_workers):
                            day_start = day * periods_per_day
                            day_end = (day + 1) * periods_per_day
                            daily_work = sum(X[d, t].solution_value() for d in range(day_start, day_end))
                            if daily_work > max_daily + 0.1:
                                violations.append(f"limite_diario: Driver {t}, Day {day}: {daily_work:.1f} > {max_daily}")
                    if not violations:
                        print(f"✅ limite_diario: OK (max 9h/day)")
                
                # Validar limite_semanal (56h/semana)
                if selected_restrictions.get("limite_semanal", False):
                    periods_per_week = 96 * 7
                    max_weekly = 224
                    for week in range(num_periods // periods_per_week):
                        for t in range(limit_workers):
                            week_start = week * periods_per_week
                            week_end = (week + 1) * periods_per_week
                            weekly_work = sum(X[d, t].solution_value() for d in range(week_start, week_end))
                            if weekly_work > max_weekly + 0.1:
                                violations.append(f"limite_semanal: Driver {t}, Week {week}: {weekly_work:.1f} > {max_weekly}")
                    if not violations:
                        print(f"✅ limite_semanal: OK (max 56h/week)")
                
                # Validar repouso_diario_minimo (11h descanso)
                if selected_restrictions.get("repouso_diario_minimo", False):
                    periods_per_day = 96
                    max_work_normal = 52
                    for day in range(num_periods // periods_per_day):
                        for t in range(limit_workers):
                            day_start = day * periods_per_day
                            day_end = (day + 1) * periods_per_day
                            daily_work = sum(X[d, t].solution_value() for d in range(day_start, day_end))
                            if daily_work > 60 + 0.1:  # max_work_reduced
                                violations.append(f"repouso_diario_minimo: Driver {t}, Day {day}: {daily_work:.1f} > 60")
                    if not violations:
                        print(f"✅ repouso_diario_minimo: OK (min 11h rest/day)")
                
                # Validar pausa_45_minutos
                if selected_restrictions.get("pausa_45_minutos", False):
                    max_continuous = 18
                    window = 21
                    for t in range(limit_workers):
                        for start in range(num_periods - window + 1):
                            continuous_work = sum(X[start + p, t].solution_value() for p in range(window))
                            if continuous_work > max_continuous + 0.1:
                                violations.append(f"pausa_45_minutos: Driver {t}, Period {start}: {continuous_work:.1f} > {max_continuous}")
                    if not violations:
                        print(f"✅ pausa_45_minutos: OK (break after 4h30)")
                
                if violations:
                    print(f"⚠️  VIOLATIONS DETECTED:")
                    for v in violations[:10]:  # Mostrar apenas as primeiras 10
                        print(f"   - {v}")
                else:
                    print(f"✅ All active constraints validated successfully!")
                
                print(f"==========================\n")
            
            # Build statistics_result as list of formatted strings (matching original V9.6)
            initial_density = calculate_density(constraints_coefficients) if constraints_coefficients is not None else 0.0
            final_density = calculate_density(matrix_allocation) if matrix_allocation is not None else 0.0
            
            statistics_result.append(f"Density check: final={final_density:.4f}, threshold={densidade_aceitavel if densidade_aceitavel else 0.01:.4f}")
            
            if status == pywraplp.Solver.OPTIMAL:
                statistics_result.append(f"Model State: OPTIMAL")
            elif status == pywraplp.Solver.FEASIBLE:
                statistics_result.append(f"Model State: FEASIBLE")
            elif status == pywraplp.Solver.INFEASIBLE:
                statistics_result.append(f"Model State: INFEASIBLE")
            elif status == pywraplp.Solver.UNBOUNDED:
                statistics_result.append(f"Model State: UNBOUNDED")
            elif status == pywraplp.Solver.ABNORMAL:
                statistics_result.append(f"Model State: ABNORMAL")
            elif status == pywraplp.Solver.MODEL_INVALID:
                statistics_result.append(f"Model State: MODEL_INVALID")
            elif status == pywraplp.Solver.NOT_SOLVED:
                statistics_result.append(f"Model State: NOT_SOLVED")
            else:
                statistics_result.append("Model State: NOT_SOLVED")
            
            total_demand = sum(need)
            total_unmet = sum(U[d].solution_value() for d in range(num_periods)) if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else total_demand
            coverage = 100.0 * (total_demand - total_unmet) / max(1.0, total_demand)
            total_capacity = sum(X[d,t].solution_value() for d in range(num_periods) for t in range(limit_workers)) if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else 0
            avg_load_per_driver = total_capacity / max(1, total_active)
            
            statistics_result.append(f"Total Demand: {total_demand}")
            statistics_result.append(f"Total Unmet: {total_unmet:.1f}")
            statistics_result.append(f"Coverage (%): {coverage:.2f}")
            statistics_result.append(f"Total Active Drivers: {total_active}")
            statistics_result.append(f"Total Assigned Slots: {total_slots}")
            statistics_result.append(f"Model Type: {tipo_modelo(solver)}")
            statistics_result.append(f"Total Resolution Time: {solver.wall_time()} ms")
            statistics_result.append(f"Total Number of Iterations: {solver.iterations()}")
            statistics_result.append(f"Number of Restrictions: {solver.NumConstraints()}")
            statistics_result.append(f"Number of Variables: {solver.NumVariables()}")
            statistics_result.append(f"AVG Load per Driver: {avg_load_per_driver:.2f}")
            
            objective_value = solver.Objective().Value() if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else 0
            best_bound = solver.Objective().BestBound() if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else 0
            gap = abs(objective_value - best_bound) / max(1.0, abs(objective_value)) if objective_value != 0 else 0
            
            statistics_result.append(f"Objective value: {objective_value:.2f}")
            statistics_result.append(f"Best bound: {best_bound:.2f}")
            statistics_result.append(f"Gap (%): {gap * 100:.2f}")
            statistics_result.append(f"Solve time (s): {solver.wall_time()/1000:.1f}")
            statistics_result.append(f"Stopped by time/gap: {gap <= 0.10 or solver.wall_time() >= time_limit_seconds * 1000}")

        return (solver, status, total_active, total_slots, workers_schedule, tasks_schedule, driving_hours_per_driver, constraints_coefficients, initial_density, final_density, statistics_result, [], [], matrix_allocation, {"stdout": stdout_buffer.getvalue(), "stderr": stderr_buffer.getvalue()})

# ==============================================================================
#  WRAPPER
# ==============================================================================

def run_solver_with_mode(
    mode, need, variable_type, constraints_coefficients, selected_restrictions,
    solver_param_type, densidade_aceitavel, limit_workers, limit_iteration,
    limit_level_relaxation, max_demands_per_driver, tolerance_demands, penalty,
    swap_rows=None, multiply_row=None, add_multiple_rows=None,
    radio_selection_object=None, enable_symmetry_breaking=False, time_limit_seconds=300,
    max_lns_iterations=5, enable_assignment_maximization=True
):
    initial_allocation = greedy_initial_allocation(need, limit_workers, max_demands_per_driver, assignment_scorer)
    if mode == "Exact": return solve_shift_schedule(solver_param_type, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows, multiply_row, add_multiple_rows, densidade_aceitavel, limit_workers, limit_iteration, limit_level_relaxation, max_demands_per_driver, tolerance_demands, penalty, initial_allocation, None, radio_selection_object, mode="Exact", enable_symmetry_breaking=enable_symmetry_breaking, time_limit_seconds=time_limit_seconds, enable_assignment_maximization=enable_assignment_maximization)
    if mode == "Heuristic":
        try:
            print(f"\n=== HEURISTIC MODE DEBUG ===")
            print(f"initial_allocation shape: {initial_allocation.shape}")
            print(f"initial_allocation sum: {initial_allocation.sum()}")
            
            total_slots = int(initial_allocation.sum())
            total_active = int(np.sum(np.any(initial_allocation > 0, axis=0)))
            print(f"total_slots: {total_slots}, total_active: {total_active}")
            
            ws, alloc = normalize_solver_outputs(need, None, initial_allocation)
            print(f"After normalize: ws={type(ws)}, alloc={type(alloc) if alloc is not None else None}")
            
            # Safety check: if alloc is None, use initial_allocation
            if alloc is None:
                print("WARNING: alloc is None, using initial_allocation")
                alloc = initial_allocation
            
            tasks_schedule = list(np.sum(alloc, axis=1)) if alloc is not None else [0] * len(need)
            print(f"tasks_schedule length: {len(tasks_schedule)}")
            print(f"=== HEURISTIC MODE SUCCESS ===\n")
            
            # statistics_result must be a list of strings in format "Key: Value"
            statistics_result = [
                f"Model Type: Heuristic (Greedy)",
                f"Model State: HEURISTIC",
                f"Total Active Drivers: {total_active}",
                f"Total Assigned Slots: {total_slots}",
                f"Execution Time: < 1s (greedy heuristic)"
            ]
            
            return (None, "HEURISTIC", total_active, total_slots, ws, tasks_schedule, {}, constraints_coefficients, None, None, statistics_result, [], [], alloc, {"stdout": "", "stderr": ""})
        except Exception as e:
            print(f"\n=== HEURISTIC MODE ERROR ===")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"=== END ERROR ===\n")
            raise
    if mode == "LNS":
        # Calculate time budget per iteration to keep LNS faster than Exact
        sub_limit = max(5, int(time_limit_seconds / (max_lns_iterations + 1)))
        
        best_sol, info = run_lns(initial_allocation, need, variable_type, constraints_coefficients, selected_restrictions, solver_param_type, limit_workers, limit_iteration, limit_level_relaxation, max_demands_per_driver, tolerance_demands, penalty, max_lns_iterations, solve_shift_schedule, neighborhood_scorer, time_limit_per_iteration=sub_limit, radio_selection_object=radio_selection_object)
        best_sol = best_sol if best_sol is not None else initial_allocation
        ws, alloc = normalize_solver_outputs(need, None, best_sol)
        # Safety check: if alloc is None, use best_sol
        if alloc is None:
            alloc = best_sol
        total_active, total_slots = int(np.sum(np.any(alloc > 0, axis=0))), int(alloc.sum())
        tasks_schedule = list(np.sum(alloc, axis=1)) if alloc is not None else [0] * len(need)
        
        # statistics_result must be a list of strings in format "Key: Value"
        statistics_result = [
            f"Model Type: Matheuristic (LNS + MILP)",
            f"Model State: LNS",
            f"Total Active Drivers: {total_active}",
            f"Total Assigned Slots: {total_slots}",
            f"LNS Iterations: {info.get('iterations', 'N/A') if info else 'N/A'}"
        ]
        
        return (None, "LNS", total_active, total_slots, ws, tasks_schedule, {}, constraints_coefficients, None, None, statistics_result, [], info.get('history', []), alloc, {"stdout": "", "stderr": ""})
    return (None, "NOT_SOLVED", 0, 0, None, None, {}, None, None, None, ["Model State: NOT_SOLVED"], [], [], None, {})

def parse_statistics_list(statistics_result):
    """
    Converte statistics_result (list[str]) em dict com chaves normalizadas.
    Aceita formatos:
      - "Key: Value"
      - "Key = Value"
      - "Key value" (fallback simples)
    """
    out = {}

    for line in statistics_result:
        if not isinstance(line, str):
            continue
        s = line.strip()
        if not s:
            continue

        # separadores comuns
        if ":" in s:
            k, v = s.split(":", 1)
        elif "=" in s:
            k, v = s.split("=", 1)
        else:
            # fallback: tenta dividir no primeiro espaço
            parts = s.split(None, 1)
            if len(parts) != 2:
                continue
            k, v = parts[0], parts[1]

        key = k.strip().lower()
        val = v.strip()

        # normalização de chaves principais
        key = key.replace("model state", "status_text")
        key = key.replace("total active drivers", "total_active_drivers")
        key = key.replace("total assigned slots", "total_assigned_slots")
        key = key.replace("total resolution time", "solve_time_ms")
        key = key.replace("total number of iterations", "iterations")
        key = key.replace("number of restrictions", "num_constraints")
        key = key.replace("number of variables", "num_vars")
        key = key.replace("objective value", "objective_value")
        key = key.replace("best bound", "best_bound")
        key = key.replace("gap (%)", "gap_percent")
        key = key.replace("solve time (s)", "solve_time_s")
        key = key.replace("stopped by time/gap", "stopped_by_limit")
        key = key.replace("density check", "density_check")

        # Conversão de tipos
        try:
            if key == "status_text":
                # Mapeia string para constante se possível, ou mantém string
                if val.upper() == "OPTIMAL":
                    out["status"] = pywraplp.Solver.OPTIMAL
                elif val.upper() == "FEASIBLE":
                    out["status"] = pywraplp.Solver.FEASIBLE
                elif val.upper() == "INFEASIBLE":
                    out["status"] = pywraplp.Solver.INFEASIBLE
                elif val.upper() == "NOT_SOLVED":
                    out["status"] = pywraplp.Solver.NOT_SOLVED
                elif val.upper() == "HEURISTIC":
                     # Caso especial para heuristics
                    out["status"] = "HEURISTIC"
                elif val.upper() == "LNS":
                    out["status"] = "LNS"
                else:
                    out["status"] = pywraplp.Solver.NOT_SOLVED
                out[key] = val

            elif key in ["total_active_drivers", "total_assigned_slots", "iterations", "num_constraints", "num_vars"]:
                out[key] = int(val)
            elif key in ["solve_time_ms", "solve_time_s", "objective_value", "best_bound", "gap_percent"]:
                 out[key] = float(val)
            else:
                 out[key] = val
        except:
            out[key] = val
            
    # Computa gap se não existir
    if "gap" not in out:
        if "gap_percent" in out:
             out["gap"] = out["gap_percent"] / 100.0
        elif "best_bound" in out and "objective_value" in out:
            # gap = |obj - bound| / |obj| roughly
            try:
                obj = out["objective_value"]
                bnd = out["best_bound"]
                if abs(obj) > 1e-9:
                    out["gap"] = abs(obj - bnd) / abs(obj)
                else:
                    out["gap"] = 0.0
            except:
                out["gap"] = None
    
    return out
