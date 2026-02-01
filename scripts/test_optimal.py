
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.solver.engine import run_solver_with_mode

def test_optimal_conditions():
    # Base demand: ultra-small (2 hours)
    demand = [1, 2, 4, 3, 2, 2, 1, 1]  # 8 periods
    
    restrictions = {
        "cobertura_necessidade": True, 
        "limite_diario": True, 
        "pausa_45_minutos": True
    }
    
    print("Testing SMALL SCALE for OPTIMAL status...")
    res = run_solver_with_mode(
        mode="Exact", 
        need=demand, 
        variable_type="Integer",
        constraints_coefficients={}, 
        selected_restrictions=restrictions,
        solver_param_type="SCIP", 
        densidade_aceitavel=0.01, 
        limit_workers=10,  # Small pool
        limit_iteration=0, 
        limit_level_relaxation=0, 
        cap_tasks_per_driver_per_slot=6,
        tolerance_demands=0.01, 
        penalty=5000, 
        time_limit_seconds=30
    )
    
    status = res[1]
    stats = res[10]
    print(f"\nResulting Status: {status}")
    for line in stats:
        if "Model State" in line or "Total Active" in line:
            print(f"  {line}")

    if "OPTIMAL" in str(status):
        print("\n✅ Success! The exactly optimal solution WAS found for a small-scale problem.")
    else:
        print("\nℹ️  Still in FALLBACK/FEASIBLE. Complexity is indeed high.")

if __name__ == "__main__":
    test_optimal_conditions()
