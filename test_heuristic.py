"""
Test Heuristic mode in isolation to identify crash cause
"""
import sys
sys.path.insert(0, 'c:/project/article')

from src.solver.engine import run_solver_with_mode
import numpy as np

# Simple test case
need = [10, 15, 20, 15, 10, 5, 8, 12] * 12  # 96 periods (1 day)
print(f"Need: {len(need)} periods")

try:
    result = run_solver_with_mode(
        mode="Heuristic",
        need=need,
        variable_type="Integer",
        constraints_coefficients=None,
        selected_restrictions={
            "cobertura_necessidade": True,
            "pausa_45_minutos": False,
            "limite_diario": False,
        },
        solver_param_type="SCIP",
        densidade_aceitavel=0.01,
        limit_workers=200,
        limit_iteration=100,
        limit_level_relaxation=10,
        max_demands_per_driver=36,
        tolerance_demands=0.01,
        penalty=0.01,
        swap_rows=None,
        multiply_row=None,
        add_multiple_rows=None,
        radio_selection_object="Minimize Total Number of Drivers",
        enable_symmetry_breaking=False,
        time_limit_seconds=60
    )
    
    print("\n✅ SUCCESS!")
    print(f"Status: {result[1]}")
    print(f"Total Active: {result[2]}")
    print(f"Total Slots: {result[3]}")
    print(f"Workers Schedule: {result[4][:10] if result[4] else None}...")
    print(f"Tasks Schedule: {result[5][:10] if result[5] else None}...")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
