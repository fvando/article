import pytest
import numpy as np
from src.solver.heuristic import greedy_initial_allocation

def test_greedy_initial_allocation_basic():
    need = [2] * 20 # Short horizon
    limit_workers = 5
    max_demands_per_driver = 10
    
    alloc = greedy_initial_allocation(
        need=need,
        limit_workers=limit_workers,
        max_demands_per_driver=max_demands_per_driver
    )
    
    assert isinstance(alloc, np.ndarray)
    assert alloc.shape == (20, limit_workers)
    
    # Verify no driver exceeds max demands
    driver_loads = np.sum(alloc, axis=0)
    assert np.all(driver_loads <= max_demands_per_driver)

def test_greedy_allocation_hint_format():
    # Verify that the output is strictly 0s and 1s (integers)
    need = [1] * 10
    limit_workers = 2
    
    alloc = greedy_initial_allocation(need, limit_workers, 5)
    
    unique_vals = np.unique(alloc)
    for val in unique_vals:
        assert val in [0, 1]
    
    assert alloc.dtype == int or alloc.dtype == np.int32 or alloc.dtype == np.int64
