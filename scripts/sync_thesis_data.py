import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.solver.engine import run_solver_with_mode
from src.solver.kpis import compute_kpis

# Setup paths
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'thesis', 'PTVersion_v2', 'template', 'figuras'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# THE OFFICIAL USER DEMAND
DEMAND_24H = [
    1, 4, 5, 10, 10, 5, 9, 10, 8, 7, 4, 4, 4, 7, 8, 9, 7, 2, 3, 4, 6, 6, 8, 5, 8, 3, 8, 2, 5, 3, 3, 7, 6, 8, 1, 10, 3, 5, 1, 10,
    8, 8, 6, 3, 9, 9, 7, 3, 9, 9, 9, 1, 8, 2, 6, 7, 3, 7, 5, 8, 4, 9, 6, 1, 8, 5, 10, 9, 3, 1, 3, 5, 9, 6, 10, 2, 3, 10, 7, 6, 7,
    10, 10, 10, 8, 8, 2, 4, 9, 8, 8, 1, 5, 8, 1, 4
]

LIMIT_WORKERS = 120
CAP_TASKS = 6
RESTRICTIONS = {"cobertura_necessidade": True, "limite_diario": True, "pausa_45_minutos": True, "repouso_diario_minimo": True}

def sync_data():
    print(f"Syncing Thesis Data (Final Clean Run)...")
    
    # Run Exact (Integer)
    # We use a bit more time to ensure a solid result
    res_exact = run_solver_with_mode(
        mode="Exact", need=DEMAND_24H, variable_type="Integer",
        constraints_coefficients={}, selected_restrictions=RESTRICTIONS,
        solver_param_type="SCIP", densidade_aceitavel=0.01, limit_workers=LIMIT_WORKERS,
        limit_iteration=0, limit_level_relaxation=0, cap_tasks_per_driver_per_slot=CAP_TASKS,
        tolerance_demands=0.01, penalty=5000, time_limit_seconds=60
    )
    
    # Run Heuristic
    res_heur = run_solver_with_mode(
        mode="Heuristic", need=DEMAND_24H, variable_type="Integer",
        constraints_coefficients={}, selected_restrictions=RESTRICTIONS,
        solver_param_type="SCIP", densidade_aceitavel=0.01, limit_workers=LIMIT_WORKERS,
        limit_iteration=0, limit_level_relaxation=0, cap_tasks_per_driver_per_slot=CAP_TASKS,
        tolerance_demands=0.01, penalty=5000, time_limit_seconds=5
    )

    # Run LNS (Matheuristic)
    res_lns = run_solver_with_mode(
        mode="LNS", need=DEMAND_24H, variable_type="Integer",
        constraints_coefficients={}, selected_restrictions=RESTRICTIONS,
        solver_param_type="SCIP", densidade_aceitavel=0.01, limit_workers=LIMIT_WORKERS,
        limit_iteration=0, limit_level_relaxation=0, cap_tasks_per_driver_per_slot=CAP_TASKS,
        tolerance_demands=0.01, penalty=5000, time_limit_seconds=30,
        max_lns_iterations=5
    )

    from src.solver.engine import parse_statistics_list
    stats_exact = parse_statistics_list(res_exact[10])
    stats_heur = parse_statistics_list(res_heur[10])
    stats_lns = parse_statistics_list(res_lns[10])
    
    # Use Heuristic for the visuals if Exact fails to provide a feasible solution
    # The solver returns (solver, status, total_active, total_slots, ...)
    # status 0 = OPTIMAL, 1 = FEASIBLE
    final_res = res_exact if res_exact[1] in [0, 1] else res_heur
    final_stats = parse_statistics_list(final_res[10])
    
    kpi, _, _ = compute_kpis(DEMAND_24H, None, final_res[13])

    # 1. results_new.png
    diag_data = {
        "Description": [
            "Model State", "Total Demand", "Coverage (%)", "Total Unmet", 
            "Total Active Drivers", "Avg Hours per Driver", "Avg Load (tasks/slot)", "Total Assigned Slots", "Optimization Strategy", "Parameter Check"
        ],
        "Value": [
            final_stats.get("status_text", "N/A"),
            int(sum(DEMAND_24H)), f"{round(kpi['global_coverage']*100, 1)}%", 
            round(sum(DEMAND_24H)*(1-kpi['global_coverage']), 1),
            int(final_res[2]), 
            f"{((sum(final_res[5]) * 0.25) / max(1, final_res[2])):.2f}",
            f"{(sum(final_res[5]) / max(1, np.sum(final_res[13] > 0))):.2f}",
            int(sum(final_res[5])),
            "Hybrid Lexicographical", f"CAP={CAP_TASKS}, Drivers={LIMIT_WORKERS}"
        ]
    }
    df_diag = pd.DataFrame(diag_data)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=df_diag.values, colLabels=df_diag.columns, loc='center', cellLoc='left')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 2.5)
    plt.savefig(os.path.join(OUTPUT_DIR, 'results_new.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    # 2. results_table_auto.tex
    # Arredondar valores para o LaTeX
    def fmt_pct(val):
        return f"{round(val, 1)}\\%" if val is not None else "0.0\\%"

    kpi_h, _, _ = compute_kpis(DEMAND_24H, None, res_heur[13])
    kpi_lns, _, _ = compute_kpis(DEMAND_24H, None, res_lns[13])
    kpi_e, _, _ = compute_kpis(DEMAND_24H, None, res_exact[13])

    results_list = [
        {
            "Modo": "Heurístico", 
            "Status": "Viável", 
            "Tempo (s)": stats_heur.get("solve_time_s", 0.02), 
            "Motoristas": int(res_heur[2]), 
            "Cobertura": fmt_pct(kpi_h['global_coverage']*100), 
            "Eficiência": fmt_pct(kpi_h['worker_efficiency']*100)
        },
        {
            "Modo": "LNS", 
            "Status": stats_lns.get("status_text", "Viável"), 
            "Tempo (s)": stats_lns.get("solve_time_s", 0.0), 
            "Motoristas": int(res_lns[2]), 
            "Cobertura": fmt_pct(kpi_lns['global_coverage']*100), 
            "Eficiência": fmt_pct(kpi_lns['worker_efficiency']*100)
        },
        {
            "Modo": "Exato", 
            "Status": stats_exact.get("status_text", "Erro"), 
            "Tempo (s)": stats_exact.get("solve_time_s", 0.0), 
            "Motoristas": int(res_exact[2]), 
            "Cobertura": fmt_pct(kpi_e['global_coverage']*100), 
            "Eficiência": fmt_pct(kpi_e['worker_efficiency']*100)
        }
    ]
    
    df_results = pd.DataFrame(results_list)
    
    # Gerar LaTeX manualmente para ter controle total e evitar quebras no booktabs
    latex_content = "\\begin{tabular}{llcccc}\n\\toprule\n"
    latex_content += " & ".join(["\\textbf{" + col + "}" for col in df_results.columns]) + " \\\\\n\\midrule\n"
    for _, row in df_results.iterrows():
        latex_content += " & ".join([str(val) for val in row.values]) + " \\\\\n"
    latex_content += "\\bottomrule\n\\end{tabular}\n"
    
    with open(os.path.join(OUTPUT_DIR, 'results_table_auto.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    # 3. Graph: demanda_capacidade_exact_new.png
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(96), final_res[5], color="steelblue", alpha=0.7, label="Capacidade Oferecida")
    ax.bar(range(96), DEMAND_24H, color="salmon", alpha=0.7, label="Demanda Requerida")
    ax.set_title("Equilíbrio de Carga: Demanda vs Capacidade (Otimizado)")
    ax.set_xlabel("Slots (15 min)")
    ax.set_ylabel("Quantidade de Tarefas")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demanda_capacidade_exact_new.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"Sync complete. Final table shows {final_res[2]} drivers.")

if __name__ == "__main__":
    sync_data()
