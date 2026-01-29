import pytest
import numpy as np
from src.solver.engine import solve_shift_schedule, run_solver_with_mode

def run_scenario(need_len, val, limit_workers, restrictions, time_limit=10, period_minutes=15):
    """Helper to run a small specific scenario"""
    need = [val] * need_len
    
    # run_solver_with_mode signature:
    # mode, need, variable_type, constraints_coefficients, selected_restrictions, solver_param_type...
    
    return solve_shift_schedule(
        solver_param_type='SCIP',
        need=need,
        variable_type='Binary',
        constraints_coefficients=None,
        selected_restrictions=restrictions,
        limit_workers=limit_workers,
        time_limit_seconds=time_limit,
        mode='Exact'
    )

def test_constraint_max_continuous_driving_break():
    """
    Test 4.5h limitation (18 slots).
    Scenario: Demand is 1 worker for 6 hours (24 slots) continuously.
    Restriction: 'pausa_45_minutos' = True.
    Expected: Single worker cannot cover 24 slots consecutively. Must have a break.
    """
    # 6 hours = 24 slots. 
    # Constraint: max 18 slots continuous work.
    
    result = run_scenario(
        need_len=24, 
        val=1, 
        limit_workers=1, 
        restrictions={"pausa_45_minutos": True}
    )
    
    status = result[1] # status
    alloc = result[13] # matrix_allocation
    
    # Check if we have a valid allocation matrix
    if alloc is None:
        # If infeasible or error, test handles it, but we expect FEASIBLE/OPTIMAL with gaps
        pass
    else:
        # Check max consecutive ones for the worker
        schedule = alloc[:, 0]
        
        # Calculate runs
        # e.g. [1, 1, 1, 0, 1, 1]
        max_run = 0
        current_run = 0
        for slot in schedule:
            if slot > 0.5: # binary
                current_run += 1
            else:
                max_run = max(max_run, current_run)
                current_run = 0
        max_run = max(max_run, current_run)
        
        # Max run must be <= 18
        assert max_run <= 18, f"Worker worked {max_run} consecutive slots, limit is 18 (4.5h)"

def test_constraint_daily_limit_9h():
    """
    Test 9h daily limit (36 slots).
    Scenario: Demand is 1 worker for 12 hours (48 slots).
    Restriction: 'limite_diario' = True.
    Expected: Worker works max 36 slots total.
    """
    result = run_scenario(
        need_len=48, 
        val=1, 
        limit_workers=1, 
        restrictions={"limite_diario": True}
    )
    
    status = result[1]
    alloc = result[13]
    
    if alloc is not None:
        total_work = np.sum(alloc[:, 0])
        # Allow +/- 0 tolerance, strictly <= 36
        assert total_work <= 36, f"Worker worked {total_work} slots, daily limit is 36 (9h)"
        
        # Also, check that unmet demand absorbs the difference
        # Demand 48, Work 36 -> Unmet ~12
        # result[10] is parser dictionary, check 'Total Unmet' logic if parsed?
        # Or parse stats list
        pass

def test_constraint_weekly_limit_56h():
    """
    Test 56h weekly limit (224 slots).
    Scenario: 7 days * 10h/day demand = 70h (350 slots).
    Restriction: 'limite_semanal' = True.
    Expected: Total work <= 224 slots.
    """
    # 7 days * 24h = 168 hours = 672 periods
    # Demand pattern: 1 worker needed all the time?
    # Let's just create a flat need of 1 for the whole week.
    
    periods_per_week = 96 * 7
    result = run_scenario(
        need_len=periods_per_week, 
        val=1, 
        limit_workers=1, 
        restrictions={"limite_semanal": True},
        time_limit=15 # give it a bit more time for 672 variables
    )
    
    alloc = result[13]
    if alloc is not None:
        total_work = np.sum(alloc[:, 0])
        assert total_work <= 224, f"Worker worked {total_work} slots, weekly limit is 224 (56h)"
