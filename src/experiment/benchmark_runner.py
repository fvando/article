import sys
import os
import json
import time
import gc
import numpy as np
import multiprocessing
from typing import Dict, Any, List

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.solver.engine import run_solver_with_mode
from src.app.simulation import get_profiles, get_restrictions, prepare_optimization_inputs
from src.solver.kpis import compute_kpis

RESULTS_FILE = os.path.join(current_dir, "benchmark_results_v2.json")

# USER PROVIDED CUSTOM DATA (24h)
DEMAND_24H_RAW = [
    1, 4, 5, 10, 10, 5, 9, 10, 8, 7, 4, 4, 4, 7, 8, 9, 7, 2, 3, 4, 6, 6, 8, 5, 8, 3, 8, 2, 5, 3, 3, 7, 6, 8, 1, 10, 3, 5, 1, 10,
    8, 8, 6, 3, 9, 9, 7, 3, 9, 9, 9, 1, 8, 2, 6, 7, 3, 7, 5, 8, 4, 9, 6, 1, 8, 5, 10, 9, 3, 1, 3, 5, 9, 6, 10, 2, 3, 10, 7, 6, 7,
    10, 10, 10, 8, 8, 2, 4, 9, 8, 8, 1, 5, 8, 1, 4
]

# Ensure exactly 96 slots (pad or trim if needed, but looks correct)
if len(DEMAND_24H_RAW) < 96:
    DEMAND_24H_RAW.extend([1] * (96 - len(DEMAND_24H_RAW)))
elif len(DEMAND_24H_RAW) > 96:
    DEMAND_24H_RAW = DEMAND_24H_RAW[:96]

LIMIT_WORKERS = 120
CAP_TASKS = 6

def generate_extended_need(base_pattern: List[int], days: int, jitter: float = 0.1, seed: int = 42) -> List[int]:
    """Extends a daily pattern to multiple days with optional noise."""
    np.random.seed(seed)
    full_need = []
    
    for d in range(days):
        # Add slight variation per day
        day_noise = np.random.uniform(1.0 - jitter, 1.0 + jitter, size=len(base_pattern))
        day_need = [max(1, int(val * noise)) for val, noise in zip(base_pattern, day_noise)]
        full_need.extend(day_need)
        
    return full_need

def solver_worker(queue, kwargs):
    """Worker function to run the solver in a separate process."""
    try:
        # Redirect stdout/stderr to avoid cluttering main process output (optional but good for clean logs)
        # For now, let's keep it visible for debugging
        res = run_solver_with_mode(**kwargs)
        queue.put(("SUCCESS", res))
    except Exception as e:
        queue.put(("ERROR", str(e)))

def run_solver_safe(timeout, **kwargs):
    """Runs the solver in a separate process with a timeout."""
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=solver_worker, args=(queue, kwargs))
    p.start()
    
    p.join(timeout)
    
    if p.is_alive():
        print(f"    !!! Solver timed out after {timeout}s. Terminating process...")
        p.terminate()
        p.join()
        return "TIMEOUT", None
    
    if not queue.empty():
        status, payload = queue.get()
        return status, payload
    else:
        # Process died without returning (e.g. segfault / OOM)
        return "CRASHED", None

def run_benchmark():
    print("Starting Benchmark Suite (Version 3 - Fast)...")
    
    profiles = get_profiles()
    scenarios = [
        # SCENARIO 1: 24 Hours
        {
            "name": "24h",
            "profile_key": "P2_STRICT_24H", 
            "hours": 24,
            "period_minutes": 15,
            "need": DEMAND_24H_RAW
        },
        # SCENARIO 2: 7 Days
        {
            "name": "7d",
            "profile_key": "W1_WEEKLY_CORE",
            "hours": 168,
            "period_minutes": 15,
            "need": generate_extended_need(DEMAND_24H_RAW, 7)
        },
        # SCENARIO 3: 15 Days
        {
            "name": "15d",
            "profile_key": "B1_BIWEEKLY_CORE", 
            "hours": 360,
            "period_minutes": 15,
            "need": generate_extended_need(DEMAND_24H_RAW, 15)
        }
    ]

    modes = ["Heuristic", "LNS", "Exact"]

    
    # Load existing results to support resume
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {RESULTS_FILE}")
        except Exception as e:
            print(f"Error loading existing results: {e}. Starting fresh.")
            results = []
    else:
        results = []

    for scen in scenarios:
        print(f"\n--- Running Scenario: {scen['name']} ({scen['hours']}h) ---")
        
        profile = profiles[scen['profile_key']]
        params = profile["params"]
        
        demand_str = ", ".join(map(str, scen['need']))
        
        # Setup Restrictions
        all_restrictions_keys = [
            "cobertura_necessidade", "limite_diario", "pausa_45_minutos", 
            "divisao_pausa1530", "divisao_pausa3015", "repouso_diario_minimo", 
            "repouso_diario_reduzido", "limite_semanal", "limite_quinzenal", 
            "repouso_semanal", "repouso_quinzenal", "descanso_apos_trabalho"
        ]
        
        selected_restrictions = {k: False for k in all_restrictions_keys}
        if profile["restrictions_on"] == "ALL":
             for k in selected_restrictions: selected_restrictions[k] = True
        else:
             for k in profile["restrictions_on"]: selected_restrictions[k] = True

        # Prepare Cache
        inputs = prepare_optimization_inputs(
            need_input=demand_str,
            total_hours=scen['hours'],
            period_minutes=scen['period_minutes'],
            restrictions=get_restrictions(),
            selected_restrictions=selected_restrictions,
            update_cache=True, 
            cache_path=f"benchmark_constraints_{scen['name']}.json"
        )
        
        constraints_coefficients = inputs["constraints_coefficients"]
        need_processed = inputs["need"]
        
        for mode in modes:
            # Check if this scenario+mode is already done
            already_done = any(r['scenario'] == scen['name'] and r['mode'] == mode for r in results)
            if already_done:
                print(f"  > Skipping Mode: {mode} (Already completed)...")
                continue
            # SKIP EXACT FOR 15 DAYS - DISABLED BY USER REQUEST
            # if mode == "Exact" and scen['name'] == "15d":
            #     print(f"  > Skipping Mode: {mode} (Too expensive for 15d)...")
            #     continue

            print(f"  > Executing Mode: {mode}...")
            
            timeout = 300 
            if mode == "Exact":
                if scen['name'] == "7d":
                    timeout = 60 # STRICT 60s timeout for 7d Exact to prove scalability limit quickly
                elif scen['name'] == "24h":
                     timeout = 180 # 3 min for 24h
                     
            # Increase timeout for 15d LNS to allow it to run longer before we assume it's stuck,
            # but still catch hard crashes.
            if scen['name'] == "15d":
                timeout = 3600 # 1 hour for 15d (LNS or Exact)

            start_time = time.time()
            
            try:
                # Call Solver safely in subprocess
                solver_kwargs = {
                    "mode": mode,
                    "need": need_processed,
                    "variable_type": "Binary",
                    "constraints_coefficients": constraints_coefficients,
                    "selected_restrictions": selected_restrictions,
                    "solver_param_type": "SCIP",
                    "densidade_aceitavel": 0.01,
                    "limit_workers": LIMIT_WORKERS,
                    "limit_iteration": 0,
                    "limit_level_relaxation": 0,
                    "max_demands_per_driver": CAP_TASKS,
                    "tolerance_demands": params.get("tolerance_coverage", 1.0),
                    "penalty": params.get("penalty_unmet", 500),
                    "radio_selection_object": "Minimize Total Number of Drivers",
                    "enable_symmetry_breaking": False,
                    "time_limit_seconds": timeout, # Solver internal timeout
                    "max_lns_iterations": 8, 
                    "enable_assignment_maximization": True
                }

                # Add a buffer to the process timeout so the solver handles its own timeout first if possible
                process_timeout = timeout + 30 
                
                status_code, res_payload = run_solver_safe(process_timeout, **solver_kwargs)
                
                elapsed = time.time() - start_time

                if status_code == "SUCCESS":
                    (solver, status, total_active, total_slots, ws, ts, _, _, _, _, _, _, _, alloc, logs) = res_payload
                    
                    # KPIs
                    kpis, _, _ = compute_kpis(need_processed, workers_schedule=ws, matrix_allocation=alloc)
                    
                    # Status string
                    status_clean = str(status)
                    if status_clean == "0": status_clean = "OPTIMAL"
                    elif status_clean == "1": status_clean = "FEASIBLE"
                    elif status_clean == "2": status_clean = "INFEASIBLE"
                    elif status_clean == "3": status_clean = "MODEL_INVALID"
                    elif status_clean == "4": status_clean = "NOT_SOLVED"
                    
                    # Special handling for NOT_SOLVED/INFEASIBLE in 7d
                    if mode == "Exact" and scen['name'] == "7d" and status_clean in ["NOT_SOLVED", "INFEASIBLE", "4", "2"]:
                        if elapsed >= timeout * 0.9:
                            status_clean = "TIMEOUT"
                    
                    result_entry = {
                        "scenario": scen['name'],
                        "mode": mode,
                        "time_s": round(elapsed, 2),
                        "active_drivers": total_active,
                        "avg_hours_dr": round((total_slots * 0.25) / max(1, total_active), 2),
                        "avg_load_slot": round(total_slots / max(1, np.sum(alloc > 0)), 2),
                        "total_assigned_slots": total_slots,
                        "coverage_pct": kpis.get("global_coverage", 0) * 100,
                        "efficiency_pct": kpis.get("worker_efficiency", 0) * 100,
                        "operational_risk": kpis.get("operational_risk", 0),
                        "status": status_clean
                    }
                    print(f"    Done. Time: {elapsed:.2f}s, Drivers: {total_active}, Cov: {result_entry['coverage_pct']:.1f}%, Status: {status_clean}")

                else:
                    # Handle CRASHED or TIMEOUT from wrapper
                    print(f"    FAILURE in {mode}: {status_code}")
                    
                    # FALLBACK LOGIC: Capture "Current Solution" (Heuristic) if LNS fails
                    fallback_entry = None
                    if mode == "LNS" and status_code in ["TIMEOUT", "CRASHED"]:
                        # Look for Heuristic result in existing results
                        for r in results:
                            if r["scenario"] == scen["name"] and r["mode"] == "Heuristic":
                                print(f"    -> Using Heuristic result as fallback for LNS.")
                                fallback_entry = r.copy()
                                fallback_entry["mode"] = "LNS"
                                fallback_entry["status"] = f"{status_code}_FALLBACK"
                                fallback_entry["time_s"] = round(elapsed, 2)
                                fallback_entry["error"] = "Time limit reached, returned best known (Heuristic)"
                                break
                    
                    if fallback_entry:
                        result_entry = fallback_entry
                    else:
                        result_entry = {
                            "scenario": scen['name'],
                            "mode": mode,
                            "time_s": round(elapsed, 2),
                            "status": status_code,
                            "error": "Process terminated abnormally" if status_code == "CRASHED" else "Process timed out"
                        }

                results.append(result_entry)
                
            except Exception as e:
                print(f"    ERROR in {mode}: {e}")
                results.append({
                    "scenario": scen['name'],
                    "mode": mode,
                    "error": str(e)
                })

            # Save Results Incrementally (after each mode)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=4)
        
        # Free memory
        gc.collect()

    print(f"\nBenchmark completed. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    run_benchmark()
