from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any

class SolverParams(BaseModel):
    solver_param_type: Literal["GLOP", "SCIP"] = "GLOP"
    variable_type: Literal["Continuous", "Binary", "Integer"] = "Integer"
    limit_workers: int = Field(default=100, ge=0)
    limit_iteration: int = Field(default=0, ge=0)
    limit_level_relaxation: float = 0.0
    cap_tasks_per_driver_per_slot: int = Field(default=1, ge=1)
    tolerance_demands: float = 0.01
    penalty: float = 0.01
    radio_selection_object: str = "Minimize Total Number of Drivers"
    mode: Literal["Exact", "Heuristic", "LNS"] = "Exact"
    enable_symmetry_breaking: bool = False

class OptimizationInput(BaseModel):
    need: List[int]
    total_hours: int
    period_minutes: int
    selected_restrictions: Dict[str, bool]
    params: SolverParams

    @field_validator("need")
    def check_need_size(cls, v):
        if not v:
            raise ValueError("Need vector cannot be empty")
        return v

class OptimizationOutput(BaseModel):
    status: str
    total_active_drivers: int
    total_assigned_slots: int
    workers_schedule: List[int]
    tasks_schedule: List[float]
    solver_stats: Dict[str, Any] = {}
    logs: Dict[str, str] = {}
