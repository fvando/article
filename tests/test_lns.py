import pytest
import numpy as np
from src.solver.engine import run_solver_with_mode

def test_lns_execution_flow():
    """
    Verify that LNS runs without errors and returns a valid schedule structure.
    """
    # Small scenario: 24h
    need = [1] * 96
    limit_workers = 2
    
    # Run in LNS mode
    result = run_solver_with_mode(
        mode="LNS",
        need=need,
        variable_type="Binary",
        constraints_coefficients=None,
        selected_restrictions={},
        solver_param_type="SCIP",
        densidade_aceitavel=0.01,
        limit_workers=limit_workers,
        limit_iteration=0,
        limit_level_relaxation=0,
        max_demands_per_driver=5,
        tolerance_demands=0.01,
        penalty=0.01,
        time_limit_seconds=10,
        max_lns_iterations=2  # Limited iterations for speed
    )
    
    # Check return structure
    # (solver, status, total_active, total_slots, ws, task_sched, driving_hours, constr, init_d, final_d, stats, ...)
    # However, LNS wrapper returns: (None, "LNS", total_active, total_slots, ws, tasks_schedule, ..., alloc, ...)
    
    status = result[1]
    assert status == "LNS"
    
    alloc = result[13] # matrix_allocation index
    assert alloc is not None
    assert alloc.shape == (96, limit_workers)
    
    stats = result[10] # statistics list
    # Ensure statistics mentions LNS
    assert any("LNS" in str(line) for line in stats)

def test_lns_improvement_potential():
    # Only verify that LNS logic doesn't crash on slightly larger instances
    # and produces non-empty results
    need = [2] * 48 # 12h
    limit_workers = 5
    
    result = run_solver_with_mode(
        mode="LNS",
        need=need,
        variable_type="Binary",
        constraints_coefficients=None,
        selected_restrictions={},
        solver_param_type="SCIP",
        densidade_aceitavel=0.01,
        limit_workers=limit_workers,
        limit_iteration=0,
        limit_level_relaxation=0,
        max_demands_per_driver=5,
        tolerance_demands=0.01,
        penalty=0.01,
        time_limit_seconds=5,
        max_lns_iterations=1
    )
    
    alloc = result[13]
    assert np.sum(alloc) > 0 # Should assert some work is assigned
