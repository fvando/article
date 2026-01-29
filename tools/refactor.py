import os

src_path = "src/app/simulation.py"
# src_path = "src/vis/start.py" # DEBUG

if not os.path.exists(src_path):
    print(f"File not found: {src_path}")
    exit(1)

with open(src_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def find_line(lines, substring, start_idx=0):
    for i in range(start_idx, len(lines)):
        if substring in lines[i]:
            return i
    return -1

# Block 1: interpret_solver_result
idx1_start = find_line(lines, "def interpret_solver_result")
idx1_end = find_line(lines, "def plot_painel_esforco_operacional", idx1_start)

# Block 2: save_data ... solve_shift_schedule
idx2_start = find_line(lines, "def save_data", idx1_end if idx1_end != -1 else 0)
idx2_end = find_line(lines, "def verifica_divisao_pausa", idx2_start)

print(f"Block 1 (interpret): {idx1_start} to {idx1_end}")
print(f"Block 2 (solver): {idx2_start} to {idx2_end}")

if idx1_start == -1 or idx1_end == -1 or idx2_start == -1 or idx2_end == -1:
    print("ERROR: Could not find all markers. Aborting to avoid corruption.")
    exit(1)

# Imports block
imports = """
# ============================================================
# IMPORTS FROM ENGINE (REFATURAÇÃO)
# ============================================================
from src.solver.engine import (
    run_solver_with_mode,
    interpret_solver_result,
    calculate_density,
    save_data,
    load_data,
    tipo_modelo,
    format_lp_output,
    rebuild_allocation_from_schedule,
    normalize_solver_outputs,
    get_solver_status_description
)
# ============================================================

"""

# Insert imports after line 70 (approx after original imports)
import_point = 70

# Construct new content
# 0..import_point
# IMPORTS
# import_point..idx1_start
# (skip block 1)
# idx1_end..idx2_start
# (skip block 2)
# idx2_end..EOF

# Note: Python slices are [start:end] (exclusive)
new_lines = (
    lines[:import_point] + 
    [imports] + 
    lines[import_point:idx1_start] + 
    lines[idx1_end:idx2_start] + 
    lines[idx2_end:]
)

backup_path = src_path + ".bak"
with open(backup_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print(f"Backup saved to {backup_path}")

with open(src_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Refactoring applied successfully.")
