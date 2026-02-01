
import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.solver.engine import run_solver_with_mode

def compare_pause_impact():
    # Official 24h demand
    demand = [1, 4, 5, 10, 10, 5, 9, 10, 8, 7, 4, 4, 4, 7, 8, 9, 7, 2, 3, 4, 6, 6, 8, 5, 8, 3, 8, 2, 5, 3, 3, 7, 6, 8, 1, 10, 3, 5, 1, 10, 8, 8, 6, 3, 9, 9, 7, 3, 9, 9, 9, 1, 8, 2, 6, 7, 3, 7, 5, 8, 4, 9, 6, 1, 8, 5, 10, 9, 3, 1, 3, 5, 9, 6, 10, 2, 3, 10, 7, 6, 7, 10, 10, 10, 8, 8, 2, 4, 9, 8, 8, 1, 5, 8, 1, 4]
    
    scenarios = [
        {"name": "WITH 45-min Break", "pausa": True},
        {"name": "WITHOUT 45-min Break", "pausa": False}
    ]
    
    results = []
    
    for scen in scenarios:
        print(f"\n--- Running Scenario: {scen['name']} ---")
        restrictions = {
            "cobertura_necessidade": True, 
            "limite_diario": True, 
            "pausa_45_minutos": scen['pausa']
        }
        
        start = time.time()
        res = run_solver_with_mode(
            mode="Exact", 
            need=demand, 
            variable_type="Integer",
            constraints_coefficients={}, 
            selected_restrictions=restrictions,
            solver_param_type="SCIP",
            densidade_aceitavel=0.01,
            limit_workers=120,
            limit_iteration=0,
            limit_level_relaxation=0,
            cap_tasks_per_driver_per_slot=6,
            tolerance_demands=0.01,
            penalty=5000,
            time_limit_seconds=60  # Quick run
        )
        elapsed = time.time() - start
        
        # res[2] is total_active, res[3] is total_slots, res[1] is status
        results.append({
            "name": scen['name'],
            "status": res[1],
            "drivers": res[2],
            "time": elapsed
        })

    print("\n" + "="*40)
    print("COMPARISON RESULTS:")
    print("="*40)
    for r in results:
        print(f"{r['name']}:")
        print(f"  Status:  {r['status']}")
        print(f"  Drivers: {r['drivers']}")
        print(f"  Time:    {r['time']:.2f}s")
    print("="*40)

if __name__ == "__main__":
    compare_pause_impact()
