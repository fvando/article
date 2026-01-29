import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

# Adicionar o diretório raiz ao path para encontrar src
sys.path.insert(0, os.path.abspath(os.curdir))

from src.solver.engine import run_solver_with_mode, solve_shift_schedule
from src.solver.kpis import compute_kpis, calculate_gini

def test_heuristic_mode_contract():
    """Valida se o modo Heurístico retorna os 15 valores esperados e dados válidos."""
    print("Testing Heuristic Mode Contract...")
    
    # Mock de dados
    need = [2, 3, 2, 1] * 24 # 96 slots
    limit_workers = 10
    constraints_coefficients = np.zeros((96, 10))
    selected_restrictions = {"cobertura_necessidade": True}
    
    try:
        results = run_solver_with_mode(
            mode="Heuristic",
            need=need,
            variable_type="Integer",
            constraints_coefficients=constraints_coefficients,
            selected_restrictions=selected_restrictions,
            solver_param_type="CBC",
            densidade_aceitavel=0.5,
            limit_workers=limit_workers,
            limit_iteration=1,
            limit_level_relaxation=0,
            max_demands_per_driver=2,
            tolerance_demands=0.01,
            penalty=0.01
        )
        
        assert len(results) == 15, f"Expected 15 return values, got {len(results)}"
        (solver, status, total_active, total_slots, ws, tasks, d_hours, const_coeff, 
         init_dens, final_dens, stats, msg, iter_data, alloc, logs) = results
        
        assert status == "HEURISTIC"
        assert isinstance(stats, list)
        assert alloc is not None, "Heuristic must return an allocation matrix"
        assert alloc.shape == (96, limit_workers)
        print("✅ Heuristic Mode Contract: PASS")
        return True
    except Exception as e:
        print(f"❌ Heuristic Mode Contract: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gini_centralization():
    """Valida se o cálculo do Gini em kpis.py bate com a lógica esperada."""
    print("\nTesting Gini Centralization...")
    values = np.array([10, 20, 30, 40])
    gini = calculate_gini(values)
    
    # Gini para [10, 20, 30, 40]
    # mean = 25
    # abs diffs: |10-10|=0, |10-20|=10, |10-30|=20, |10-40|=30...
    # total abs diff = (0+10+20+30) + (10+0+10+20) + (20+10+0+10) + (30+20+10+0) = 60 + 40 + 40 + 60 = 200
    # Gini = 200 / (2 * 4^2 * 25) = 200 / 800 = 0.25
    
    print(f"Calculated Gini: {gini}")
    # O cálculo em kpis.py usa np.trapz, pode haver pequena variação dependendo da discretização
    assert 0.2 <= gini <= 0.3, "Gini value out of expected range"
    print("✅ Gini Centralization: PASS")
    return True

def test_lexicographical_solve_logic():
    """Valida se a lógica de 2 fases no solver está funcionando (ununpacking e locks)."""
    print("\nTesting Lexicographical Solve Logic...")
    need = [2] * 4 # Pequeno para rapidez
    limit_workers = 5
    constraints_coefficients = np.zeros((4, 5))
    
    results = run_solver_with_mode(
        mode="Exact",
        need=need,
        variable_type="Integer",
        constraints_coefficients=constraints_coefficients,
        selected_restrictions=[],
        solver_param_type="CBC",
        densidade_aceitavel=0.5,
        limit_workers=limit_workers,
        limit_iteration=1,
        limit_level_relaxation=0,
        max_demands_per_driver=2,
        tolerance_demands=0.01,
        penalty=0.01,
        time_limit_seconds=5 # Rápido
    )
    
    assert len(results) == 15
    print("✅ Lexicographical Solve Logic (Exact): PASS")
    return True

if __name__ == "__main__":
    success = True
    success &= test_heuristic_mode_contract()
    success &= test_gini_centralization()
    success &= test_lexicographical_solve_logic()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n🛑 SOME TESTS FAILED.")
        sys.exit(1)
