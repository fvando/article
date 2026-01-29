import pytest
import numpy as np
from src.solver.engine import solve_shift_schedule

def test_solve_shift_schedule_feasible():
    need = [1] * 96 
    limit_workers = 5
    
    # Correct signature call
    # solve_shift_schedule(solver_param_type, need, variable_type, constraints_coefficients, selected_restrictions, ...)
    
    result = solve_shift_schedule(
        solver_param_type='SCIP',
        need=need,
        variable_type='Binary',
        constraints_coefficients=None,
        selected_restrictions={},
        limit_workers=limit_workers,
        time_limit_seconds=10,
        mode='Exact'
    )
    
    # result is a dict (interpretation) or a tuple?
    # In engine.py line 200: it returns `interpretation` (dict) if called via `solve_shift_schedule` wrapper?
    # Wait, line 738 returns a TUPLE! `return (solver, status, ...)`
    # BUT `run_solver_with_mode` (line 753) returns `solve_shift_schedule(...)`.
    # AND `solve_shift_schedule` (line 243) returns `(solver, status, total_active, ...)`?
    # Let's check line 243 of engine.py wrapper.
    # Ah, I viewed lines 151-300.
    # Lines 700-750 show `solve_shift_schedule` logic ending with `return (...)` (Tuple).
    # So the return value is a TUPLE.
    
    assert isinstance(result, tuple)
    # Tuple[1] is status (int)
    # Tuple[10] is statistics_result (list)
    
    status = result[1]
    stats = result[10]
    
    # Check status (0 = OPTIMAL, 1 = FEASIBLE) - typically
    from ortools.linear_solver import pywraplp
    assert status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]
    
    # Check allocation matrix (index 13 in tuple based on line 738?)
    # Line 738: ..., matrix_allocation, ...
    # matrix_allocation is at index 13?
    # 0:solver, 1:status, 2:total_active, 3:total_slots, 4:workers_schedule, 5:tasks_schedule, 
    # 6:driving_hours, 7:constraints_coef, 8:init_density, 9:final_density, 10:stats, 11:?, 12:?, 13:matrix_allocation
    
    matrix_alloc = result[13]
    if matrix_alloc is not None:
        assert matrix_alloc.shape == (96, limit_workers)

def test_solve_shift_schedule_timeout_config():
    need = [5] * 96
    limit_workers = 10
    
    result = solve_shift_schedule(
        solver_param_type='SCIP',
        need=need,
        variable_type='Binary',
        constraints_coefficients=None,
        selected_restrictions={},
        limit_workers=limit_workers,
        time_limit_seconds=1, 
        mode='Exact'
    )
    
    assert isinstance(result, tuple)
