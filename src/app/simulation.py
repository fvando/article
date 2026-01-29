# Caminho absoluto para a pasta "simulator"

# Altura padrão graficos
# import streamlit as st
# from pathlib import Path


# # ------------------------------------------------------------
# # Secrets / Config
# # ------------------------------------------------------------
# DEMO_MODE = st.secrets.get("DEMO_MODE", True)
# MAX_TIME = st.secrets.get("MAX_TIME_SECONDS", 30)

import matplotlib
matplotlib.use('Agg')

# 🔒 CRITICAL: Threading lock for matplotlib operations
# Matplotlib is not thread-safe, and Streamlit runs in multiple threads.
# This lock serializes all matplotlib operations to prevent race conditions.
import threading
_MPL_LOCK = threading.Lock()

def safe_render_figure(fig, container=None):
    """
    Safely render a matplotlib figure to Streamlit using st.image().
    This avoids asyncio event loop conflicts that occur with st.pyplot().
    
    Args:
        fig: matplotlib Figure object
        container: optional Streamlit container (column, expander, etc). If None, uses st directly.
    """
    import io
    import matplotlib.pyplot as plt
    import streamlit as st
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    
    target = container if container is not None else st
    target.image(buf, width='stretch')
    
    buf.close()
    plt.close(fig)

FIG_HEIGHT = 4.5

import os
import sys
from typing import Any, List, Optional, Tuple

BASE_SIMULATOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


from src.vis.ui import apply_corporate_style
from src.core.i18n import t

# apply_corporate_style() # Invoked by main.py to avoid duplicate widgets


# Garante que o Python reconhece "ml/" como pacote
if BASE_SIMULATOR_DIR not in sys.path:
    sys.path.append(BASE_SIMULATOR_DIR)

import html
import math
import streamlit as st

# Language Selector handled in src.vis.ui


import numpy as np
import seaborn as sns
import pandas as pd
import json
from math import ceil
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import io, contextlib

# from heuristic import greedy_initial_allocation
# from lns import run_lns
from src.ml.ml_guidance import assignment_scorer, neighborhood_scorer
# from dataset_builder import build_f1_rows, build_f2_rows, save_datasets
from src.ml.trainer import train_all_models
import traceback
import time
# from kpis import compute_kpis
# from charts import (
#     plot_heatmap_safety,
#     plot_heatmap_coverage,
#     plot_kpi_radar,
# )

from src.solver.heuristic import greedy_initial_allocation
from src.solver.kpis import compute_kpis, calculate_gini
from src.vis.charts import (
    plot_heatmap_safety,
    plot_heatmap_coverage,
    plot_kpi_radar,
)
from src.solver.lns import run_lns
from src.solver.dataset_builder import build_f1_rows, build_f2_rows, save_datasets

import html
# Habilitar a largura total da página
# st.set_page_config(layout="wide") # Removido (controlado por main.py)

#---------------------------------

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

def check_shift_violations(matrix_allocation, min_shift_periods=16):
    """Checks for SHIFT_MIN violations in a matrix allocation."""
    if matrix_allocation is None:
        return 0
    violations = 0
    try:
        import numpy as np
        num_periods, num_workers = matrix_allocation.shape
        for t in range(num_workers):
            schedule = matrix_allocation[:, t]
            # Find contiguous work blocks
            is_working = schedule > 0
            # diff detects starts (1) and ends (-1)
            diff = np.diff(is_working.astype(int), prepend=0, append=0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                duration = e - s
                if duration < min_shift_periods:
                    violations += 1
    except Exception as e:
        print(f"Error checking shift violations: {e}")
        pass
    return violations
# ============================================================

# Graficos
#--------------------------------
def get_objective_details():
    return {
        t("math_minimize_drivers"): [
            {
                "Term": t("math_unmet_penalty"),
                "Notation": r"$\alpha \sum_{d \in D} U_d$",
                "Role": t("math_role_primary"),
                "Explanation": t("math_unmet_desc")
            },
            {
                "Term": t("math_active_penalty"),
                "Notation": r"$\beta \sum_{t \in T} Z_t$",
                "Role": t("math_role_secondary"),
                "Explanation": t("math_active_desc")
            },
            {
                "Term": t("math_task_reward"),
                "Notation": r"$-\gamma \sum_{d \in D} \sum_{t \in T} X_{d,t}$",
                "Role": t("math_role_tie"),
                "Explanation": t("math_task_desc")
            },
            {
                "Term": t("math_presence_min"),
                "Notation": r"$\omega_1 \sum_{d,t} Y_{d,t}$",
                "Role": t("math_role_phase2"),
                "Explanation": t("math_presence_desc")
            },
            {
                "Term": t("math_overload"),
                "Notation": r"$\omega_2 \sum_t \max(0, L_t - L^{soft})$",
                "Role": t("math_role_phase2"),
                "Explanation": t("math_overload_desc")
            },
            {
                "Term": t("math_smoothing"),
                "Notation": r"$\omega_3 L_{\max}$",
                "Role": t("math_role_phase2"),
                "Explanation": t("math_smoothing_desc")
            }
        ]
    }

def render_math_table(rows):
    table_md = f"""
| {t('lbl_row')} | Notation | {t('col_algo')} | Explain |
|------|----------|------|-------------|
"""
    for r in rows:
        table_md += (
            f"| {r['Term']} "
            f"| $${r['Notation']}$$ "
            f"| {r['Role']} "
            f"| {r['Explanation']} |\n"
        )

    st.markdown(table_md, unsafe_allow_html=True)

def get_lex_objectives():
    return {
        t("math_minimize_drivers"): [
            {
                "priority": 1,
                "name": t("lex_unmet"),
                "formula": r"\min \sum_{d \in D} U_d",
                "description": t("lex_unmet_desc")
            },
            {
                "priority": 2,
                "name": t("lex_driver_min"),
                "formula": r"\min \sum_{t \in T} Z_t",
                "description": t("lex_driver_desc")
            },
            {
                "priority": 3,
                "name": t("lex_util"),
                "formula": r"\max \sum_{d \in D} \sum_{t \in T} X_{d,t}",
                "description": t("lex_util_desc")
            },
            {
                "priority": 4,
                "name": t("lex_workload"),
                "formula": r"\min \left(\sum Y_{d,t} + \sum \max(0, L_t - L^{soft}) + L_{\max}\right)",
                "description": t("lex_workload_desc")
            }
        ]
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_timeseries_demand_served(
    need,
    tasks_schedule,
    slot_minutes: int = 15
):
    import numpy as np
    import pandas as pd

    demand = np.asarray(need, dtype=float)
    served = np.asarray(tasks_schedule, dtype=float)

    slots = np.arange(len(demand))
    time_hours = slots * (slot_minutes / 60.0)

    gap = np.maximum(0.0, demand - served)

    return pd.DataFrame({
        "slot": slots,
        "time_hours": time_hours,
        "demand_tasks": demand,
        "served_tasks": served,
        "gap_tasks": gap,
    })


def build_timeseries_capacity_utilization(
    workers_schedule,
    tasks_schedule,
    cap_tasks_per_driver_per_slot: int,
    slot_minutes: int = 15
):
    import numpy as np
    import pandas as pd

    workers = np.asarray(workers_schedule, dtype=float)
    served  = np.asarray(tasks_schedule, dtype=float)

    slots = np.arange(len(workers))
    time_hours = slots * (slot_minutes / 60.0)

    capacity = workers * cap_tasks_per_driver_per_slot

    utilization = np.divide(
        served, capacity,
        out=np.zeros_like(served),
        where=capacity > 0
    )

    return pd.DataFrame({
        "slot": slots,
        "time_hours": time_hours,
        "capacity_tasks": capacity,
        "served_tasks": served,
        "utilization": utilization,
    })

def build_timeseries_gap(
    need,
    tasks_schedule,
    slot_minutes: int = 15
):
    import numpy as np
    import pandas as pd

    demand = np.asarray(need, dtype=float)
    served = np.asarray(tasks_schedule, dtype=float)

    slots = np.arange(len(demand))
    time_hours = slots * (slot_minutes / 60.0)

    gap = np.maximum(0.0, demand - served)

    return pd.DataFrame({
        "slot": slots,
        "time_hours": time_hours,
        "gap_tasks": gap,
    })

def plot_demand_gap(
    demanda,
    tasks_schedule,
    slot_minutes: int = 15,
    title: str = None
):
    """
    Gráfico do gap (déficit) por período.
    Mostra explicitamente onde a demanda não foi atendida.
    """
    from matplotlib.figure import Figure

    if title is None:
        title = "Demand Deficit per Period"

    demanda = np.asarray(demanda, dtype=float).reshape(-1)
    served  = np.asarray(tasks_schedule, dtype=float).reshape(-1)

    if len(demanda) != len(served):
        raise ValueError("demanda e tasks_schedule devem ter o mesmo tamanho")

    slots = np.arange(len(demanda))
    time_hours = slots * (slot_minutes / 60.0)

    gap = np.maximum(0.0, demanda - served)

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.bar(
        time_hours,
        gap,
        width=slot_minutes / 60.0 * 0.9,
        alpha=0.85
    )

    ax.set_title(title)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Deficit (tasks)")

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Métricas-chave no gráfico
    total_gap = gap.sum()
    max_gap   = gap.max()
    slots_gap = int((gap > 0).sum())

    ax.text(
        0.01, 0.98,
        f"Slots with deficit: {slots_gap}\n"
        f"Total deficit: {total_gap:.1f}\n"
        f"Max deficit: {max_gap:.1f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    return fig


def plot_demand_vs_served(
    demanda,
    tasks_schedule,
    slot_minutes: int = 15,
    title: str = None
):
    from matplotlib.figure import Figure
    if title is None:
        title = t("chart_demand_served_title")

    demanda = np.asarray(demanda, dtype=float).reshape(-1)
    served  = np.asarray(tasks_schedule, dtype=float).reshape(-1)

    if len(demanda) != len(served):
        raise ValueError("demanda e tasks_schedule devem ter o mesmo tamanho")

    slots = np.arange(len(demanda))
    time_hours = slots * (slot_minutes / 60.0)

    deficit = np.maximum(0.0, demanda - served)
    viol = deficit > 1e-9

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.plot(time_hours, demanda, linewidth=2, label=t("chart_demand_label"))
    ax.plot(time_hours, served,  linewidth=2, linestyle="--", label=t("chart_served_label"))

    # marca violações
    ax.scatter(time_hours[viol], demanda[viol], s=30, label=t("chart_deficit_label"), zorder=5)

    ax.set_title(title)
    ax.set_xlabel(t("axis_time_hours"))
    ax.set_ylabel(t("axis_tasks"))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")

    # texto informativo
    ax.text(
        0.01, 0.98,
        f"{t('chart_gap_slots')}: {int(viol.sum())}\n"
        f"{t('chart_gap_total')}: {deficit.sum():.1f}\n"
        f"{t('chart_gap_max')}: {deficit.max():.1f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    return fig

def plot_capacity_vs_utilization(
    workers_schedule,
    tasks_schedule,
    cap_tasks_per_driver_per_slot: int,
    slot_minutes: int = 15,
    title: str = None
):
    from matplotlib.figure import Figure
    if title is None:
        title = t("chart_cap_util_title")

    workers = np.asarray(workers_schedule, dtype=float).reshape(-1)
    served  = np.asarray(tasks_schedule, dtype=float).reshape(-1)

    if len(workers) != len(served):
        raise ValueError("workers_schedule e tasks_schedule devem ter o mesmo tamanho")

    slots = np.arange(len(workers))
    time_hours = slots * (slot_minutes / 60.0)

    capacity_tasks = workers * float(cap_tasks_per_driver_per_slot)

    utilization = np.divide(
        served, capacity_tasks,
        out=np.zeros_like(served, dtype=float),
        where=capacity_tasks > 1e-12
    )

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax1 = fig.subplots()

    # eixo esquerdo
    ax1.plot(time_hours, capacity_tasks, linewidth=2, label=t("chart_cap_alloc_label"))
    ax1.plot(time_hours, served, linewidth=2, linestyle="--", label=t("chart_served_label"))
    ax1.set_xlabel(t("axis_time_hours"))
    ax1.set_ylabel(t("axis_tasks"))
    ax1.grid(True, linestyle="--", alpha=0.35)

    # eixo direito
    ax2 = ax1.twinx()
    ax2.plot(time_hours, utilization, linewidth=2, linestyle=":", label=t("chart_util_label"), color="orange")
    ax2.set_ylabel(t("chart_util_yaxis"))
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, linestyle="--", alpha=0.5, color="red")

    # legenda combinada
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    ax1.set_title(title)

    return fig



def plot_demand_curve(
    need,
    slot_minutes: int = 15,
    title: str = None
):
    """
    Gráfico da demanda por slot (input do modelo).

    Eixo X: tempo (slots ou horas)
    Eixo Y: demanda requerida
    """
    from matplotlib.figure import Figure

    if title is None:
        title = t("chart_demand_curve_title")

    demanda = np.asarray(need, dtype=float)

    slots = np.arange(len(demanda))
    horas = slots * (slot_minutes / 60.0)

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.plot(
        horas,
        demanda,
        linewidth=2,
        color="steelblue"
    )

    ax.set_xlabel(t("axis_time_hours"))
    ax.set_ylabel(t("chart_demand_req_label"))
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.4)

    return fig

def plot_demand_coverage(
    demanda,
    workers_schedule,
    slot_minutes: int = 15,
    title: str = None
):
    """
    Gráfico de cobertura:
    - Demanda requerida
    - Motoristas alocados (capacidade)
    """

    if title is None:
        title = t("chart_cov_title")

    import numpy as np
    import matplotlib.pyplot as plt

    demanda = np.asarray(demanda, dtype=float)
    workers_schedule = np.asarray(workers_schedule, dtype=float)

    if len(demanda) != len(workers_schedule):
        raise ValueError("demanda e workers_schedule devem ter o mesmo comprimento")

    slots = np.arange(len(demanda))
    horas = slots * (slot_minutes / 60.0)

    fig, ax = plt.subplots(figsize=(14, FIG_HEIGHT))

    ax.plot(
        horas,
        demanda,
        label=t("chart_demand_req_label"),
        linewidth=2
    )

    ax.plot(
        horas,
        workers_schedule,
        label=t("chart_drivers_alloc_label"),
        linewidth=2
    )

    # Marca violações
    viol = workers_schedule < demanda
    ax.scatter(
        horas[viol],
        demanda[viol],
        color="red",
        s=30,
        label=t("chart_cov_deficit_label"),
        zorder=5
    )

    ax.set_xlabel(t("axis_time_hours"))
    ax.set_ylabel(t("axis_qty"))
    ax.set_title(title)

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    return fig


def plot_panel_a_driving(driving_hours_active):
    from matplotlib.figure import Figure
    x = np.sort(np.asarray(driving_hours_active))
    y = np.arange(1, len(x)+1) / len(x)

    fig = Figure(figsize=(14, FIG_HEIGHT))
    axes = fig.subplots(1, 2)

    # ECDF
    axes[0].plot(x, y, linewidth=2)
    axes[0].axvline(9.0, linestyle="--", color="red")
    axes[0].set_title(t("chart_driv_ecdf_title"))
    axes[0].set_xlabel(t("chart_driv_xaxis"))
    axes[0].set_ylabel(t("axis_prob"))
    axes[0].grid(alpha=0.3)

    # Boxplot
    axes[1].boxplot(x, vert=True)
    axes[1].axhline(9.0, linestyle="--", color="red")
    axes[1].set_title(t("chart_driv_box_title"))
    axes[1].set_ylabel(t("axis_hours"))
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig

def plot_panel_b_shift(shift_hours_active, SHIFT_MAX=13):
    from matplotlib.figure import Figure
    x = np.sort(np.asarray(shift_hours_active))
    y = np.arange(1, len(x)+1) / len(x)

    fig = Figure(figsize=(14, FIG_HEIGHT))
    axes = fig.subplots(1, 2)

    # ECDF
    axes[0].plot(x, y, linewidth=2)
    axes[0].axvline(SHIFT_MAX, linestyle="--", color="red")
    axes[0].set_title(t("chart_shift_ecdf_title"))
    axes[0].set_xlabel(t("chart_shift_xaxis"))
    axes[0].set_ylabel(t("axis_prob"))
    axes[0].grid(alpha=0.3)

    # Boxplot
    axes[1].boxplot(x, vert=True)
    axes[1].axhline(SHIFT_MAX, linestyle="--", color="red")
    axes[1].set_title(t("chart_shift_box_title"))
    axes[1].set_ylabel(t("axis_hours"))
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig

def compute_coverage_kpis(
    demanda,
    tasks_served_per_slot,
):
    import numpy as np

    demanda = np.asarray(demanda, dtype=float)
    atendido = np.asarray(tasks_served_per_slot, dtype=float)

    # déficit e excesso reais (baseados em X)
    deficit_per_slot = np.maximum(0.0, demanda - atendido)
    excess_per_slot  = np.maximum(0.0, atendido - demanda)

    slots_with_deficit = int(np.sum(deficit_per_slot > 1e-9))
    total_uncovered    = float(np.sum(deficit_per_slot))
    max_deficit        = float(np.max(deficit_per_slot)) if len(deficit_per_slot) else 0.0
    total_excess       = float(np.sum(excess_per_slot))

    # métricas percentuais
    total_demand = float(np.sum(demanda))
    fully_covered_pct = 100.0 * (len(demanda) - slots_with_deficit) / max(1, len(demanda))
    weighted_coverage_pct = (
        100.0 * (total_demand - total_uncovered) / max(1.0, total_demand)
    )

    return {
        "slots_with_deficit": slots_with_deficit,
        "total_uncovered": total_uncovered,
        "max_deficit": max_deficit,
        "total_excess": total_excess,
        "fully_covered_pct": fully_covered_pct,
        "weighted_coverage_pct": weighted_coverage_pct,
    }

def plot_gap_chart(demanda, workers_schedule):
    """
    Gap = Motoristas - Demanda
    Negativo  → Violação
    Positivo  → Folga
    """
    from matplotlib.figure import Figure

    df = pd.DataFrame({
        "Slot": np.arange(1, len(demanda) + 1),
        "Demanda": demanda,
        "Motoristas": workers_schedule
    })

    df["Gap"] = df["Motoristas"] - df["Demanda"]

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.axhline(0, color="black", linewidth=1)
    ax.bar(
        df["Slot"],
        df["Gap"],
        color=df["Gap"].apply(lambda x: "green" if x >= 0 else "red"),
        alpha=0.8
    )

    ax.set_title(t("chart_gap_cov_title"))
    ax.set_xlabel(t("axis_slot"))
    ax.set_ylabel(t("chart_gap_cov_yaxis"))

    # Métricas auxiliares
    violations = (df["Gap"] < 0).sum()
    min_gap = df["Gap"].min()

    ax.text(
        0.01, 0.95,
        f"{t('chart_gap_violations')}: {violations}\n{t('chart_gap_worst')}: {min_gap}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    return fig

def plot_demand_capacity_utilization(
    demanda,
    workers_schedule,                 # presença (Y agregada)
    tasks_schedule,                   # <<< atendimento real (X agregada)
    cap_tasks_per_driver_per_slot
):
    from matplotlib.figure import Figure

    demanda = np.asarray(demanda, dtype=float).reshape(-1)
    workers_schedule = np.asarray(workers_schedule, dtype=float).reshape(-1)
    
    n = len(demanda)

    if len(workers_schedule) != n:
        raise ValueError("workers_schedule length mismatch")

    if len(tasks_schedule) != n:
        raise ValueError("tasks_schedule length mismatch")


    slots = np.arange(len(demanda))

    # Capacidade REAL (derivada de Y)
    capacidade_por_slot = workers_schedule * cap_tasks_per_driver_per_slot

    # Utilização REAL (derivada de X)
    utilizacao = np.divide(
        tasks_schedule,
        capacidade_por_slot,
        out=np.zeros_like(tasks_schedule, dtype=float),
        where=capacidade_por_slot > 0
    )

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax1 = fig.subplots()

    ax1.plot(slots, demanda, label="Demanda (tarefas)", linewidth=2)
    ax1.plot(
        slots,
        capacidade_por_slot,
        label="Capacidade Alocada (tarefas)",
        linewidth=2,
        alpha=0.9
    )
    ax1.plot(
        slots,
        tasks_schedule,
        label="Atendimento Real",
        linewidth=2,
        linestyle="--"
    )

    ax1.set_xlabel(t("axis_slot"))
    ax1.set_ylabel(t("axis_tasks"))
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        slots,
        utilizacao,
        label=t("chart_util_cap_label"),
        linestyle=":",
        linewidth=2,
        color="orange"
    )
    ax2.set_ylabel(t("chart_util_yaxis"))
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, linestyle="--", alpha=0.5, color="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(t("chart_dem_cap_util_title"))

    return fig

def plot_demand_vs_capacity(
    demanda,
    workers_schedule,                 # presença (Y agregada)
    tasks_schedule,                   # atendimento real (X agregado)
    cap_tasks_per_driver_per_slot
):
    from matplotlib.figure import Figure
    demanda = np.asarray(demanda, dtype=float)
    workers_schedule = np.asarray(workers_schedule, dtype=float)
    tasks_schedule = np.asarray(tasks_schedule, dtype=float)

    capacidade = workers_schedule * cap_tasks_per_driver_per_slot
    slots = np.arange(len(demanda))

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.plot(slots, demanda, label=t("chart_demand_label"), linewidth=2)
    ax.plot(slots, capacidade, label=t("chart_cap_alloc_label"), linewidth=2)
    ax.plot(slots, tasks_schedule, label=t("chart_served_label"), linewidth=2, linestyle="--")

    ax.set_xlabel(t("axis_slot"))
    ax.set_ylabel(t("axis_tasks"))
    ax.set_title(t("chart_dem_cap_util_title"))
    ax.legend()
    ax.grid(alpha=0.3)

    return fig

def plot_capacity_utilization(demanda, matrix_allocation, cap_tasks_per_driver_per_slot):
    from matplotlib.figure import Figure
    demanda = np.asarray(demanda, dtype=float)
    matrix_allocation = np.asarray(matrix_allocation, dtype=int)

    motoristas_por_slot = matrix_allocation.sum(axis=1)
    capacidade = motoristas_por_slot * cap_tasks_per_driver_per_slot
    demanda_atendida = np.minimum(demanda, capacidade)

    utilizacao = np.divide(
        demanda_atendida,
        capacidade,
        out=np.zeros_like(demanda),
        where=capacidade > 0
    )

    slots = np.arange(len(demanda))

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.plot(slots, utilizacao, linestyle="--", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Período (slot)")
    ax.set_ylabel("Utilização")
    ax.set_title("Utilização da Capacidade Alocada")
    ax.grid(alpha=0.3)

    return fig

def plot_demand_vs_drivers_line(demanda, workers_schedule):
    from matplotlib.figure import Figure
    df = pd.DataFrame({
        "Slot": np.arange(1, len(demanda) + 1),
        "Demanda": demanda,
        "Motoristas": workers_schedule
    })

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.plot(df["Slot"], df["Demanda"], label="Demanda", linewidth=2)
    ax.plot(df["Slot"], df["Motoristas"], label="Motoristas Alocados", linewidth=2)

    ax.set_title("Demanda vs Motoristas Alocados")
    ax.set_xlabel("Período (slot)")
    ax.set_ylabel("Quantidade")

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    return fig

def plot_operational_effort_per_driver(
    matrix_allocation,
    slot_minutes: int = 15,
    daily_limit_hours: float = 9.0,
    title: str = None,
):
    """
    Gráfico 1 — Esforço Operacional por Motorista
    """
    from matplotlib.figure import Figure

    if title is None:
        title = t("chart_op_effort_title")

    if matrix_allocation is None:
        return None

    mat = np.asarray(matrix_allocation, dtype=int)

    if mat.ndim != 2 or mat.size == 0:
        return None

    # -----------------------------
    # Cálculos
    # -----------------------------
    slot_hours = slot_minutes / 60.0
    slots_per_driver = mat.sum(axis=0)
    active_mask = slots_per_driver > 0
    hours_per_driver = slots_per_driver[active_mask] * slot_hours    
    avg_hours = float(hours_per_driver.mean()) if len(hours_per_driver) else 0.0

    # -----------------------------
    # Gráfico
    # -----------------------------
    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.bar(
        range(1, len(hours_per_driver) + 1),
        hours_per_driver,
        alpha=0.85,
        label=t("legend_worked_hours")
    )

    ax.axhline(
        daily_limit_hours,
        color="red",        
        linestyle="--",
        linewidth=2,
        label=f"{t('legend_limit_daily')} ({daily_limit_hours}h)"
    )

    ax.axhline(
        avg_hours,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"{t('legend_avg_observed')} ({avg_hours:.2f}h)"
    )

    ax.set_xlabel(t("axis_driver"))
    ax.set_ylabel(t("axis_hours_worked_horizon"))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Retorna o fig e os dados para o st.caption
    resumo = {
        "total_assigned_slots": int(slots_per_driver.sum()),
        "total_hours": float(slots_per_driver.sum() * slot_hours),
        "avg_hours": avg_hours,
        "slot_minutes": slot_minutes,
        "daily_limit_hours": daily_limit_hours
    }

    return fig, resumo

    # -----------------------------
    # Texto interpretativo
    # -----------------------------
    st.caption(
        f"""
        Total de slots atribuídos: **{total_assigned_slots}**  
        Duração por slot: **{slot_minutes} minutos**  
        Total de horas trabalhadas: **{total_hours:.2f} h**

        Em média, cada motorista trabalhou aproximadamente **{avg_hours:.2f} h**.
        Este valor é coerente com o limite diário de **{daily_limit_hours} h**,
        considerando a flexibilidade residual decorrente da não ativação do
        repouso diário mínimo neste cenário.

        ⚠️ **Importante:** este indicador representa **esforço operacional**.
        Ele **não é o objetivo do modelo**, mas um resultado emergente utilizado
        para análise de eficiência e validação regulatória.
        """
    )

import numpy as np
import matplotlib.pyplot as plt

def plot_operational_effort_per_driver_active(
    matrix_allocation: np.ndarray,
    slot_minutes: int = 15,
    limite_diario_horas: float = 9.0,
):
    """
    Plota o esforço operacional considerando APENAS os motoristas ativos.
    """
    from matplotlib.figure import Figure

    if matrix_allocation is None or matrix_allocation.size == 0:
        raise ValueError(t("msg_chart_invalid_matrix"))

    # ------------------------------------------------
    # 1) Carga por motorista (em slots)
    # ------------------------------------------------
    slots_por_motorista = matrix_allocation.sum(axis=0)

    # considerar APENAS motoristas ativos
    slots_motoristas_ativos = slots_por_motorista[slots_por_motorista > 0]

    if len(slots_motoristas_ativos) == 0:
        raise ValueError("Nenhum motorista ativo na solução.")

    # ------------------------------------------------
    # 2) Conversão para horas
    # ------------------------------------------------
    horas_por_motorista = slots_motoristas_ativos * (slot_minutes / 60.0)

    total_slots_atribuidos = int(slots_motoristas_ativos.sum())
    total_horas = float(horas_por_motorista.sum())
    total_motoristas_ativos = int(len(horas_por_motorista))
    media_horas = float(horas_por_motorista.mean())

    # ------------------------------------------------
    # 3) Gráfico
    # ------------------------------------------------
    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()
    
    ax.bar(
        range(1, total_motoristas_ativos + 1),
        horas_por_motorista,
        color="steelblue",
        alpha=0.85,
        label=t("legend_assigned_load")
    )

    # Linha de referência legal
    ax.axhline(
        limite_diario_horas,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"{t('legend_limit_daily')} ({limite_diario_horas}h)"
    )

    ax.axhline(
        media_horas,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"{t('legend_avg_observed')} ({media_horas:.2f}h)"
    )
    
    ax.set_xlabel(t("axis_active_drivers_sorted"))
    ax.set_ylabel(t("axis_workload_hours"))
    ax.set_title(t("chart_op_effort_active_title"))

    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # ------------------------------------------------
    # 4) Resumo Métrica
    # ------------------------------------------------
    resumo = {
        "total_slots_atribuidos": total_slots_atribuidos,
        "total_horas": total_horas,
        "total_motoristas_ativos": total_motoristas_ativos,
        "media_horas": media_horas,
        "slot_minutes": slot_minutes,
        "limite_diario_horas": limite_diario_horas
    }
    
    return fig, resumo
    
#---------------------------------
# Interpretador de resultados
#--------------------------------

import re

def parse_statistics_list(statistics_result):
    """
    Converte statistics_result (list[str]) em dict com chaves normalizadas.
    Aceita formatos:
      - "Key: Value"
      - "Key = Value"
      - "Key value" (fallback simples)
    """
    out = {}

    for line in statistics_result:
        if not isinstance(line, str):
            continue
        s = line.strip()
        if not s:
            continue

        # separadores comuns
        if ":" in s:
            k, v = s.split(":", 1)
        elif "=" in s:
            k, v = s.split("=", 1)
        else:
            # fallback: tenta dividir no primeiro espaço
            parts = s.split(None, 1)
            if len(parts) != 2:
                continue
            k, v = parts[0], parts[1]

        key = k.strip().lower()
        val = v.strip()

        # normalização de chaves principais
        key = key.replace("model state", "status_text")
        key = key.replace("total active drivers", "total_active_drivers")
        key = key.replace("total assigned slots", "total_assigned_slots")
        key = key.replace("total resolution time", "solve_time_ms")
        key = key.replace("total number of iterations", "iterations")
        key = key.replace("number of restrictions", "num_constraints")
        key = key.replace("number of variables", "num_vars")
        key = key.replace("objective value", "objective_value")
        key = key.replace("best bound", "best_bound")
        key = key.replace("gap (%)", "gap_percent")
        key = key.replace("solve time (s)", "solve_time_s")
        key = key.replace("stopped by time/gap", "stopped_by_limit")
        key = key.replace("density check", "density_check")

        # parse de valores
        if key in ("total_active_drivers", "total_assigned_slots", "iterations", "num_constraints", "num_vars"):
            m = re.search(r"-?\d+", val)
            out[key] = int(m.group()) if m else None

        elif key in ("objective_value", "best_bound"):
            m = re.search(r"-?\d+(\.\d+)?", val.replace(",", "."))
            out[key] = float(m.group()) if m else None

        elif key == "gap_percent":
            m = re.search(r"-?\d+(\.\d+)?", val.replace(",", "."))
            out[key] = float(m.group()) if m else None

        elif key in ("solve_time_s",):
            m = re.search(r"-?\d+(\.\d+)?", val.replace(",", "."))
            out[key] = float(m.group()) if m else None

        elif key == "solve_time_ms":
            m = re.search(r"-?\d+(\.\d+)?", val.replace(",", "."))
            out[key] = float(m.group()) if m else None

        elif key == "stopped_by_limit":
            out[key] = val.lower() in ("true", "1", "yes", "sim")

        elif key == "density_check":
            # exemplo: "final=0.3542, threshold=0.0100"
            m_final = re.search(r"final\s*=\s*([0-9.]+)", val)
            m_thr = re.search(r"threshold\s*=\s*([0-9.]+)", val)
            out["density"] = float(m_final.group(1)) if m_final else None
            out["density_threshold"] = float(m_thr.group(1)) if m_thr else None
            out[key] = val

        else:
            out[key] = val

    # pós-processamento: converter texto do status em enum aproximado
    stxt = (out.get("status_text") or "").upper()
    if "OPTIMAL" in stxt:
        out["status"] = pywraplp.Solver.OPTIMAL
    elif "FEASIBLE" in stxt:
        out["status"] = pywraplp.Solver.FEASIBLE
    elif "INFEASIBLE" in stxt:
        out["status"] = pywraplp.Solver.INFEASIBLE
    else:
        out["status"] = None

    # gap em fração (0-1), se disponível
    if out.get("gap_percent") is not None:
        out["gap"] = out["gap_percent"] / 100.0
    else:
        out["gap"] = None

    return out

def plot_painel_esforco_operacional(
    horas_por_motorista: np.ndarray,
    limite_diario_horas: float = 9.0,
):
    """
    Painel integrado com:
      1) ECDF do esforço operacional
      2) Boxplot da carga por motorista
      3) Curva de Lorenz + Índice de Gini

    Parâmetros
    ----------
    horas_por_motorista : np.ndarray
        Vetor com a carga total (em horas) dos motoristas ativos
    limite_diario_horas : float
        Referência legal indicativa (ex.: 9h)

    Retorno
    -------
    fig : matplotlib.figure.Figure
    resumo : dict
        Métricas consolidadas para relatório/dissertação
    """

    from matplotlib.figure import Figure

    if horas_por_motorista is None or len(horas_por_motorista) == 0:
        raise ValueError("Vetor horas_por_motorista inválido ou vazio.")

    horas = np.sort(horas_por_motorista)
    n = len(horas)

    # ------------------------------------------------
    # Métricas
    # ------------------------------------------------
    media = float(np.mean(horas))
    mediana = float(np.median(horas))
    p90 = float(np.percentile(horas, 90))
    p95 = float(np.percentile(horas, 95))
    minimo = float(np.min(horas))
    maximo = float(np.max(horas))

    gini = calculate_gini(horas_por_motorista)

    # Vetores para a curva de Lorenz
    horas_acum = np.cumsum(horas)
    horas_acum = np.insert(horas_acum, 0, 0)
    horas_acum_norm = horas_acum / horas_acum[-1]
    proporcao_motoristas = np.linspace(0, 1, len(horas_acum_norm))

    # ------------------------------------------------
    # Layout do painel
    # ------------------------------------------------
    fig = Figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

    ax_ecdf = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1])
    ax_lorenz = fig.add_subplot(gs[1, :])

    # ------------------------------------------------
    # 1) ECDF
    # ------------------------------------------------
    ecdf = np.arange(1, n + 1) / n
    ax_ecdf.step(horas, ecdf, where="post", linewidth=2)
    ax_ecdf.axvline(limite_diario_horas, color="red", linestyle="--", linewidth=2)
    ax_ecdf.axvline(mediana, color="green", linestyle=":", linewidth=2)

    ax_ecdf.set_xlabel(t("axis_total_load_driver"))
    ax_ecdf.set_ylabel(t("axis_prob"))
    ax_ecdf.set_title(t("chart_ecdf_effort_title"))
    ax_ecdf.grid(True, linestyle=":", alpha=0.6)

    # ------------------------------------------------
    # 2) Boxplot
    # ------------------------------------------------
    ax_box.boxplot(
        horas,
        vert=True,
        widths=0.5,
        patch_artist=True
    )

    ax_box.axhline(limite_diario_horas, color="red", linestyle="--", linewidth=2)
    ax_box.set_xticks([1])
    ax_box.set_xticklabels([t("chart_active_drivers")])
    ax_box.set_ylabel(t("axis_total_load_driver"))
    ax_box.set_title(t("chart_dispersion_title"))
    ax_box.grid(axis="y", linestyle=":", alpha=0.6)

    # ------------------------------------------------
    # 3) Curva de Lorenz
    # ------------------------------------------------
    ax_lorenz.plot(
        proporcao_motoristas,
        horas_acum_norm,
        linewidth=2,
        label=t("legend_lorenz_curve")
    )

    ax_lorenz.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label=t("legend_equitable_dist")
    )

    ax_lorenz.set_xlabel(t("axis_prop_drivers"))
    ax_lorenz.set_ylabel(t("axis_prop_load"))
    ax_lorenz.set_title(f"{t('chart_equity_title')} (Gini = {gini:.3f})")
    ax_lorenz.grid(True, linestyle=":", alpha=0.6)
    ax_lorenz.legend()

    # ------------------------------------------------
    # Ajustes finais
    # ------------------------------------------------
    fig.suptitle(
        t("panel_effort_title"),
        fontsize=14,
        y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ------------------------------------------------
    # Resumo estruturado
    # ------------------------------------------------
    resumo = {
        "motoristas_ativos": n,
        "media_horas": media,
        "mediana_horas": mediana,
        "percentil_90": p90,
        "percentil_95": p95,
        "minimo_horas": minimo,
        "maximo_horas": maximo,
        "indice_gini": gini,
        "limite_referencia_horas": limite_diario_horas,
    }

    return fig, resumo

#---------------------------------
# Criador de Narrativa
#--------------------------------
def generate_narrative(interp: dict) -> str:
    m = interp["operational_metrics"]

    return (
        f"O modelo de otimização produziu uma solução viável com "
        f"{m['drivers_used']} motoristas ativos, totalizando "
        f"{m['total_hours']} horas de trabalho alocadas. "
        f"A carga média por motorista foi de {m['avg_hours_per_driver']} horas. "
        f"{interp['model_structure']} "
        f"{interp['solver_behavior']}"
    )

import matplotlib.pyplot as plt
import numpy as np

def plot_demand_vs_capacity_with_violations(df):
    """
    Gráfico Demand vs Capacity com marcação automática dos slots violados.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    # Barras: capacidade alocada
    ax.bar(
        df["Slot"],
        df["Motoristas"],
        color="steelblue",
        alpha=0.85,
        label=t("label_assigned_drivers_cap")
    )

    # Barras: demanda requerida
    ax.bar(
        df["Slot"],
        df["Demanda"],
        color="salmon",
        alpha=0.85,
        label=t("label_req_drivers_dem")
    )

    # Slots violados
    violated = df[df["Motoristas"] < df["Demanda"]]

    # Marcação visual (ponto vermelho no topo da demanda)
    ax.scatter(
        violated["Slot"],
        violated["Demanda"],
        color="red",
        s=40,
        zorder=5,
        label=t("label_dem_deficit")
    )

    ax.set_xlabel(t("axis_time_slot"))
    ax.set_ylabel(t("axis_driver"))
    ax.set_title(t("chart_dem_cap_viol_title"))

    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()

    return fig


def plot_demand_vs_capacity_2(df):
    """
    Gráfico:
      Demand vs Assigned Capacity
      com marcação visual de violação (déficit)
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    # Capacity (Assigned Drivers)
    ax.bar(
        df["Slot"],
        df["Motoristas"],
        alpha=0.85,
        label=t("label_assigned_drivers_cap")
    )

    # Demand
    ax.bar(
        df["Slot"],
        df["Demanda"],
        alpha=0.85,
        label=t("label_req_drivers_dem")
    )

    # Violations
    violated = df[df["Motoristas"] < df["Demanda"]]

    ax.scatter(
        violated["Slot"],
        violated["Demanda"],
        s=35,
        zorder=5,
        label=t("label_dem_deficit")
    )

    ax.set_ylabel(t("axis_driver"))
    ax.set_title(t("chart_dem_cap_title"))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    return fig

def plot_capacity_gap(df):
    """
    Gráfico:
      Gap entre capacidade e demanda por slot
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    gap = df["Motoristas"] - df["Demanda"]
    # colors = gap.apply(lambda x: "red" if x < 0 else "steelblue")

    ax.bar(
        df["Slot"],
        gap,
        alpha=0.85
    )

    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xlabel(t("axis_time_slot"))
    ax.set_ylabel(t("axis_gap_cap_dem"))
    ax.set_title(t("chart_cap_gap_title"))
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Worst deficit annotation
    if not df.empty:
        idx_min = gap.idxmin()
        worst = df.loc[idx_min]
        ax.annotate(
            f"{t('annot_max_deficit')} = {worst['Motoristas'] - worst['Demanda']:.1f}",
            xy=(worst["Slot"], worst["Motoristas"] - worst["Demanda"]),
            xytext=(worst["Slot"], (worst["Motoristas"] - worst["Demanda"]) - 0.5),
            arrowprops=dict(arrowstyle="->", color="red")
        )

    fig.tight_layout()
    return fig


import matplotlib.pyplot as plt

def plot_carga_motoristas_ativos_vs_inativos(ativos, inativos):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(14, FIG_HEIGHT))
    (ax1, ax2) = fig.subplots(1, 2, sharey=True)

    # Inativos
    ax1.hist(inativos, bins=1, color="lightgray", edgecolor="black")
    ax1.set_title(t("chart_inactive_drivers"))
    ax1.set_xlabel(t("axis_assigned_slots"))
    ax1.set_ylabel(t("axis_num_drivers"))
    ax1.set_xticks([0])

    # Ativos
    ax2.hist(ativos, bins=15, color="steelblue", edgecolor="black")
    ax2.set_title(t("chart_active_drivers"))
    ax2.set_xlabel(t("axis_assigned_slots"))

    fig.suptitle(t("chart_load_dist_title"))
    fig.tight_layout()

    return fig

def plot_carga_motoristas_ativos(ativos):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(14, FIG_HEIGHT))
    ax = fig.subplots()

    ax.hist(ativos, bins=15, color="steelblue", edgecolor="black")

    ax.set_xlabel(t("axis_total_slots_driver"))
    ax.set_ylabel(t("axis_num_drivers"))
    ax.set_title(t("chart_workload_dist_active_title"))

    # Estatísticas-chave
    if len(ativos) > 0:
        ax.axvline(ativos.mean(), color="orange", linestyle="--", label=f"Média = {ativos.mean():.1f}")
        ax.axvline(np.median(ativos), color="green", linestyle=":", label=f"Mediana = {np.median(ativos):.1f}")

    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig

# Inicializar flags
if "run_train_ml" not in st.session_state:
    st.session_state.run_train_ml = False

if "train_results" not in st.session_state:
    st.session_state.train_results = None

if "train_ml_result" not in st.session_state:
    st.session_state.train_ml_result = None

if "train_error" not in st.session_state:
    st.session_state.train_error = None
    
if "run_dataset_ml" not in st.session_state:
    st.session_state.run_dataset_ml = False

# Função para gerar valores aleatórios e armazená-los no cache
@st.cache_resource
def gerar_valores_aleatorios(total_hours, period_minutes):
    return np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()

def make_formula_safe(formula: str) -> str:
    """
    Converte fórmulas matemáticas para uma versão segura para Streamlit,
    evitando erros de parsing markdown como 'unmatched )'.
    Produz um texto totalmente seguro sem quebrar a UI.
    """
    if not formula:
        return ""

    safe = html.escape(formula)

    # Remove interpretações de markdown perigoso
    safe = safe.replace("|", " | ")

    # Evitar que colchetes virem links
    safe = safe.replace("[", "(").replace("]", ")")

    # Evitar inline LaTeX
    safe = safe.replace("$", "")

    # Operadores unicode → ASCII seguro
    safe = safe.replace("⋅", "*").replace("·", "*").replace("→", "->")

    # Escapar parênteses para evitar math parsing
    safe = safe.replace("(", "&#40;").replace(")", "&#41;")

    return safe

# Definindo as restrições
elementalOperations = [
    {
        "Description": "Equation Exchange",
        "Formula": r"$E_i \leftrightarrow E_j$",
        "Detalhes": "The number of workers allocated to periods that meet the valid windows must be sufficient to satisfy the minimum need of period \(i\). The operation involves swapping two lines of the system of equations.",
        "Key": "troca_equacoes"
    },
    {
        "Description": "Multiplication by Constant",
        "Formula": r"$E_i: \sum_{j} a_{ij} X_j = b_i \quad \text{becomes} \quad E_i': \sum_{j} (k \cdot a_{ij}) X_j = k \cdot b_i$",
        "Detalhes": "Multiplying a row by a constant is useful for simplifying values ​​or canceling terms, but must be done carefully to avoid an unwanted increase in matrix fill-in.",
        "Key": "multiplicacao_por_constante"
    },
    {
        "Description": "Add Multiple of One Equation to Another",
        "Formula": r"$E_2: \sum_{j} a_{2j} X_j = b_2 \quad \text{becomes} \quad E_2': \sum_{j} (a_{2j} + \alpha \cdot a_{1j}) X_j = b_2 + \alpha \cdot b_1$",
        "Detalhes": "After a maximum of 18 periods (4.5 hours), there should be a break of 3 periods (45 minutes). This operation involves adding a multiple of one row to another to manipulate the equations of the system.",
        "Key": "soma_multiplo_equacao"
    },
    {
        "Description": "Add Multiple of One Equation to Another (Automatic)",
        "Formula": r"$E_2: \sum_{j} a_{2j} X_j = b_2 \quad \text{becomes} \quad E_2': \sum_{j} (a_{2j} + \alpha \cdot a_{1j}) X_j = b_2 + \alpha \cdot b_1$",
        "Detalhes": "After a maximum of 18 periods (4.5 hours), there should be a break of 3 periods (45 minutes). This operation is done automatically by the solver to adjust the equations without manual intervention..",
        "Key": "soma_multiplo_equacao_automatica"
    }
]

# Definindo as restrições
def get_restrictions():
    return [
        {
            "Description": t("restr_cov_desc"),
            "Formula": r"$\sum_{j \in \text{valid slot}(i)} X[j] \geq \text{need}[i]$",
            "Detalhes": t("restr_cov_det"),
            "Key": "cobertura_necessidade"
        },
        {
            "Description": t("restr_daily_desc"),
            "Formula": r"$\sum_{p \in \text{day}} X[p] \leq 36p (1p=15minutes | 36p=9h)$",
            "Detalhes": t("restr_daily_det"),
            "Key": "limite_diario"
        },
        {
            "Description": t("restr_break45_desc"),
            "Formula": r"$\text{Pause} \geq 3p (1p=15minutes | 3p=45minutes)$",
            "Detalhes": t("restr_break45_det"),
            "Key": "pausa_45_minutos"
        },
        {
            "Description": t("restr_rest_daily_desc"),
            "Formula": r"$\text{Rest Period} \geq 44p (1p=15minutes |  44p=11h)$",
            "Detalhes": t("restr_rest_daily_det"),
            "Key": "repouso_diario_minimo"
        },
        {
            "Description": t("restr_rest_weekly_desc"),
            "Formula": r"$\text{Rest Period} \geq 180p \;\text{(1p = 15 minutes | 180p = 45 h)}$",
            "Detalhes": t("restr_rest_weekly_det"),
            "Key": "repouso_semanal"
        },
        {
            "Description": t("restr_rest_after_work_desc"),
            "Formula": r"$\text{6 days Work} \Rightarrow \text{Rest for One}$",
            "Detalhes": t("restr_rest_after_work_det"),
            "Key": "descanso_apos_trabalho"
        },
        {
            "Description": t("restr_weekly_limit_desc"),
            "Formula": r"$\sum_{p \in \text{week}} X[p] \leq 224p \;\text{(1p = 15 minutes | 224p = 56 h)}$",
            "Detalhes": t("restr_weekly_limit_det"),
            "Key": "limite_semanal"
        },
        {
            "Description": t("restr_biweekly_limit_desc"),
            "Formula": r"$\sum_{p \in \text{Biweekly }} X[p] \leq 360p (1p=15minutes |  360p=90h)$",
            "Detalhes": t("restr_biweekly_limit_det"),
            "Key": "limite_quinzenal"
        },
        {
            "Description": t("restr_rest_reduced_desc"),
            "Formula": r"$\geq 36p \text{ (1p=15minutes |  36p=9h | Max. 3x | 14 days)}$",
            "Detalhes": t("restr_rest_reduced_det"),
            "Key": "repouso_diario_reduzido"
        },
        {
            "Description": t("restr_rest_biweekly_desc"),
            "Formula": r"$\geq 96p (1p=15minutes, logo  96p=24h)$",
            "Detalhes": t("restr_rest_biweekly_det"),
            "Key": "repouso_quinzenal"
        }
    ]



def get_profiles():
    return {
        # =========================
        # 24h — PERFIS BÁSICOS
        # =========================
        "P0_SANITY_24H": {
            "description": t("prof_sanity"),
            "params": {
                "horizon_hours": 24,
                "slot": 15,
                "tolerance_coverage": 1.00,
                "penalty_unmet": 500,
                "max_demands_per_driver": 1,
                "limit_workers": 100,
                "limit_iteration": 0,
                "limit_level_relaxation": 0.0,
                "acceptable_density": 0.01
            },
            "restrictions_on": [
                "cobertura_necessidade",
            ],
        },

        "P1_OPERATIONAL_24H": {
            "description": t("prof_operational"),
            "params": {
                "horizon_hours": 24,
                "slot": 15,
                "tolerance_coverage": 1.00,
                "penalty_unmet": 800,
                "max_demands_per_driver": 1,
                "limit_workers": 120,
                "limit_iteration": 0,
                "limit_level_relaxation": 0.0,
                "acceptable_density": 0.01
            },
            "restrictions_on": [
                "cobertura_necessidade",
                "limite_diario",
                "pausa_45_minutos",
            ],
        },

        "P2_STRICT_24H": {
            "description": t("prof_strict"),
            "params": {
                "horizon_hours": 24,
                "slot": 15,
                "tolerance_coverage": 1.00,
                "penalty_unmet": 1500,
                "max_demands_per_driver": 1,
                "limit_workers": 180,
                "limit_iteration": 0,
                "limit_level_relaxation": 0.0,
                "acceptable_density": 0.01
            },
            "restrictions_on": [
                "cobertura_necessidade",
                "limite_diario",
                "pausa_45_minutos",
                "repouso_diario_minimo",
            ],
        },

        "P3_FLEX_24H": {
            "description": t("prof_flex"), 
            "params": {
                "horizon_hours": 24,
                "slot": 15,
                "tolerance_coverage": 0.98,
                "penalty_unmet": 2500,
                "max_demands_per_driver": 1,
                "limit_workers": 180,
                "limit_iteration": 0,
                "limit_level_relaxation": 0.0,
                "acceptable_density": 0.01            
            },
            "restrictions_on": [
                "cobertura_necessidade",
                "limite_diario",
                "pausa_45_minutos",
                "repouso_diario_minimo",
                "repouso_diario_reduzido",
            ],
        },

        # =========================
        # 7 DIAS
        # =========================

        "W1_WEEKLY_CORE": {
            "description": t("prof_weekly_core"),
            "params": {
                "horizon_hours": 168,
                "slot": 15,
                "tolerance_coverage": 0.98,
                "penalty_unmet": 3000,
                "max_demands_per_driver": 1,
                "limit_workers": 400,
                "limit_iteration": 0,
                "limit_level_relaxation": 0.0,
                "acceptable_density": 0.01            
            },
            "restrictions_on": [
                "cobertura_necessidade",
                "limite_diario",
                "pausa_45_minutos",
                "limite_semanal",
            ],
        },

        "W2_WEEKLY_STRICT": {
            "description": t("prof_weekly_strict"),
            "params": {
                "horizon_hours": 168,
                "slot": 15,
                "tolerance_coverage": 0.98,
                "penalty_unmet": 4000,
                "max_demands_per_driver": 1,
                "limit_workers": 500,
                "limit_iteration": 0,
                "limit_level_relaxation": 0.0,
                "acceptable_density": 0.01
            },
            "restrictions_on": [
                "cobertura_necessidade",
                "limite_diario",
                "pausa_45_minutos",
                "limite_semanal",
                "repouso_semanal",
            ],
        },

        # =========================
        # 14 DIAS
        # =========================

        "B1_BIWEEKLY_CORE": {
            "description": t("prof_biweekly"),
            "params": {
                "horizon_hours": 336,
                "slot": 15,
                "tolerance_coverage": 0.97,
                "penalty_unmet": 5000,
                "max_demands_per_driver": 1,
                "limit_workers": 800,
                "limit_iteration": 0,
                "limit_level_relaxation": 0.0,
                "acceptable_density": 0.01
            },
            "restrictions_on": [
                "cobertura_necessidade",
                "limite_diario",
                "pausa_45_minutos",
                "limite_semanal",
                "limite_quinzenal",
            ],
        },

        "B2_BIWEEKLY_STRICT": {
            "description": t("prof_biweekly"), # Using same key or new one if needed
            "params": {
                "horizon_hours": 336,
                "slot": 15,
                "tolerance_coverage": 0.97,
                "penalty_unmet": 7000,
                "max_demands_per_driver": 1,
                "limit_workers": 900,
                "limit_iteration": 0,
                "limit_level_relaxation": 0.0,
                "acceptable_density": 0.01
            },
            "restrictions_on": [
                "cobertura_necessidade",
                "limite_diario",
                "pausa_45_minutos",
                "limite_semanal",
                "limite_quinzenal",
                "repouso_semanal",
                "repouso_quinzenal",
                "descanso_apos_trabalho",
            ],
        },

        "D1_DEBUG_INFEASIBLE": {
            "description": "Debug Infeasible", # Keep english or add key
            "params": {
                "horizon_hours": 24,
                "slot": 15,
                "tolerance_coverage": 1.00,
                "penalty_unmet": 5000,
                "max_demands_per_driver": 1,
                "limit_workers": 1200,
                "limit_iteration": 5,
                "limit_level_relaxation": 0.05,
                "acceptable_density": 0.01
            },
            "restrictions_on": "ALL",
        },
    }

def get_elemental_operations():
    return [
        {
            "Description": t("op_swap"),
            "Formula": r"$E_i \leftrightarrow E_j$",
            "Detalhes": t("elem_swap_det"),
            "Key": "troca_equacoes"
        },
        {
            "Description": t("op_mult"),
            "Formula": r"$E_i: \sum_{j} a_{ij} X_j = b_i \quad \text{becomes} \quad E_i': \sum_{j} (k \cdot a_{ij}) X_j = k \cdot b_i$",
            "Detalhes": t("elem_mult_det"),
            "Key": "multiplicacao_por_constante"
        },
        {
            "Description": t("op_sum"),
            "Formula": r"$E_2: \sum_{j} a_{2j} X_j = b_2 \quad \text{becomes} \quad E_2': \sum_{j} (a_{2j} + \alpha \cdot a_{1j}) X_j = b_2 + \alpha \cdot b_1$",
            "Detalhes": t("elem_sum_det"),
            "Key": "soma_multiplo_equacao"
        },
        {
            "Description": f"{t('op_sum')} (Auto)",
            "Formula": r"$E_2: \sum_{j} a_{2j} X_j = b_2 \quad \text{becomes} \quad E_2': \sum_{j} (a_{2j} + \alpha \cdot a_{1j}) X_j = b_2 + \alpha \cdot b_1$",
            "Detalhes": t("elem_sum_det"),
            "Key": "soma_multiplo_equacao_automatica"
        },
    ]

import traceback
from typing import Dict, Any

def run_ml_training() -> Dict[str, Any]:
    """
    Executa o treinamento dos modelos ML f₁ e f₂.

    Returns
    -------
    dict
        {
            "success": bool,
            "results": dict | None,
            "error": str | None
        }
    """
    try:
        results = train_all_models()

        return {
            "success": True,
            "results": results,
            "error": None,
        }

    except Exception:
        return {
            "success": False,
            "results": None,
            "error": traceback.format_exc(),
        }

def render_ml_training_results(results: Dict[str, Any]) -> None:
    """
    Renderiza os resultados do treinamento ML (f₁ e f₂) na UI.
    """

    # ==========================
    # 🔹 RESULTADOS F1
    # ==========================
    f1 = results.get("f1", {})
    st.subheader(t("ml_res_f1_title"))

    if f1.get("success"):
        # Se a mensagem vier do trainer (já traduzida ou não), podemos exibir.
        # Mas aqui tínhamos hardcoded "Model f1 successfully trained!".
        # Vamos usar a chave genérica:
        st.success(t("ml_train_success_f1"))

        if f1.get("accuracy") is not None:
            st.write(f"**{t('ml_acc')}** {f1['accuracy']:.4f}")
        else:
            st.info(f"{t('ml_acc')} {t('ml_na')}")

        if "rows_used" in f1:
            st.write(f"**{t('ml_rows_used')}** {f1['rows_used']}")

        if f1.get("model_path"):
            st.write(f"**{t('ml_model_saved')}** `{f1['model_path']}`")

    else:
        st.warning(t("ml_train_fail_f1"))
        st.write(f1.get("message", t("ml_reason")))

    # ==========================
    # 🔹 RESULTADOS F2
    # ==========================
    f2 = results.get("f2", {})
    st.subheader(t("ml_res_f2_title"))

    if f2.get("success"):
        st.success(t("ml_train_success_f2"))

        if f2.get("mse") is not None:
            st.write(f"**{t('ml_mse')}** {f2['mse']:.4f}")
        else:
            st.info(f"{t('ml_mse')} {t('ml_na')}")

        if "rows_used" in f2:
            st.write(f"**{t('ml_rows_used')}** {f2['rows_used']}")

        if f2.get("model_path"):
            st.write(f"**{t('ml_model_saved')}** `{f2['model_path']}`")

    else:
        st.warning(t("ml_train_fail_f2"))
        st.write(f2.get("message", t("ml_reason")))

# Função para salvar os dados no arquivo
def verifica_divisao_pausa(i, j, num_periods):
    diff = (i - j) % num_periods
    
    # Cenário 1: Pausa de 15 minutos + 30 minutos (ex: 2h15min de condução + 30min)
    if diff == 18:  # 18 períodos de condução (4h30min)
        # Verifica se a pausa fracionada pode ser aplicada (15min + 30min ou 30min + 15min)
        if (i + 1) % num_periods == 0 or (i + 2) % num_periods == 0:
            return True
    # Cenário 2: Pausa de 30 minutos + 15 minutos (ex: 3h de condução + 15min)
    elif diff == 18:  # 18 períodos de condução (4h30min)
        if (i + 1) % num_periods == 0 or (i + 2) % num_periods == 0:
            return True

    return False

# Função que preenche a matriz de restrições de acordo com as condições e restrições
def preencher_restricoes(initial_constraints_coefficients, restrictions, selected_restrictions, num_periods, need):
    # Loop para preencher a matriz de restrições
    for i in range(num_periods):
        for j in range(num_periods):
            # Calcula a diferença cíclica (i - j) com modularidade
            diff = (i - j) % num_periods
            
            # Inicializa como sem cobertura
            initial_constraints_coefficients[i, j] = 0
            
            # Itera sobre as restrições
            for restriction in restrictions:
                # Verifica se a restrição está selecionada
                if selected_restrictions.get(restriction["Key"], False):
                    
                    # Aplique a lógica para cada restrição
                    if restriction["Key"] == "limite_diario":

                        # Primeira janela: condução inicial (0 a 18 períodos)
                        if 0 <= (i - j) % num_periods < 18:
                            initial_constraints_coefficients[i, j] = 1  # Condução permitida

                        # Segunda janela: pausa obrigatória (19 a 20 períodos) Pausa de 15 minutos
                        elif 19 <= (i - j) % num_periods < 25 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 25 <= (i - j) % num_periods < 28 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 30 <= (i - j) % num_periods < 37 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        # Terceira janela: pausa obrigatória (19 a 20 períodos) Pausa de 30 minutos
                        elif 20 <= (i - j) % num_periods < 21 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 30 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 21 <= (i - j) % num_periods < 25 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 26 <= (i - j) % num_periods < 37 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        # Quarta janela: pausa obrigatória (19 a 20 períodos) Pausa de 30 minutos
                        elif 21 <= (i - j) % num_periods < 37 and selected_restrictions.get("pausa_45_minutos", False):  # Pausa de 30 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 0 <= (i - j) % num_periods < 44 and selected_restrictions.get("repouso_diario_minimo", False):
                            # Exemplo de condição para repouso diário mínimo
                            #if 0 <= diff < 44:  # Repouso diário de 11h
                            initial_constraints_coefficients[i, j] = 1    
                        elif 0 <= (i - j) % num_periods < 36 and selected_restrictions.get("repouso_diario_reduzido", False): #restriction["Key"] == "repouso_diario_reduzido":
                            # Exemplo de condição para repouso diário reduzido
                                # if 0 <= diff < 36:  # Repouso de 9h
                            initial_constraints_coefficients[i, j] = 1                                                    
                        # Fora dos intervalos permitidos
                        else:
                            initial_constraints_coefficients[i, j] = 0  # Fora do intervalo permitido
                    elif restriction["Key"] == "repouso_semanal":
                        # Exemplo de condição para repouso semanal
                        if 0 <= diff < 180:  # Repouso semanal de 45h
                            initial_constraints_coefficients[i, j] = 1
                    elif restriction["Key"] == "repouso_quinzenal":
                        # Exemplo de condição para repouso quinzenal
                        if 0 <= diff < 96:  # Repouso quinzenal de 24h
                            initial_constraints_coefficients[i, j] = 1
                    elif restriction["Key"] == "descanso_apos_trabalho":
                        # Exemplo de condição para descanso após dias de trabalho
                        if 0 <= diff < 6:  # Após 6 dias de trabalho, descanso
                            initial_constraints_coefficients[i, j] = 1
                    elif restriction["Key"] == "limite_semanal":
                        # Limite semanal de condução (224 períodos)
                        # Verifica a soma de períodos dentro de uma semana
                        week_start = (i // 7) * 7
                        week_end = week_start + 7
                        weekly_periods = np.sum(initial_constraints_coefficients[week_start:week_end, :])
                        if weekly_periods <= 224:
                            initial_constraints_coefficients[i, j] = 1
                    elif restriction["Key"] == "limite_quinzenal":
                        # Limite quinzenal de condução (360 períodos)
                        # Verifica a soma de períodos dentro de uma quinzena
                        fortnight_start = (i // 14) * 14
                        fortnight_end = fortnight_start + 14
                        fortnight_periods = np.sum(initial_constraints_coefficients[fortnight_start:fortnight_end, :])
                        if fortnight_periods <= 360:
                            initial_constraints_coefficients[i, j] = 1

                    elif restriction["Key"] == "cobertura_necessidade":
                        # Cobertura de necessidade (verifica a necessidade de trabalhadores)
                        if np.sum(initial_constraints_coefficients[:, i]) >= need[i]:
                            initial_constraints_coefficients[i, j] = 1

                            
    return initial_constraints_coefficients

def apply_profile(profile_key, all_restrictions):
    profile = PROFILES[profile_key]

    # Resetar todas as restrições
    selected_restrictions = {k: False for k in all_restrictions}

    # Ativar restrições do perfil
    if profile["restrictions_on"] == "ALL":
        for k in selected_restrictions:
            selected_restrictions[k] = True
    else:
        for k in profile["restrictions_on"]:
            selected_restrictions[k] = True

    return profile["params"], selected_restrictions

def prepare_optimization_inputs(
    need_input: str,
    total_hours: int,
    period_minutes: int,
    restrictions: list,
    selected_restrictions: dict,
    *,
    update_cache: bool = False,
    cache_path: str = "constraints_coefficients.json",
):
    # --------------------------------------------------
    # 1) Processamento da demanda
    # --------------------------------------------------
    need = [int(x.strip()) for x in need_input.split(",") if x.strip().isdigit()]
    num_periods = len(need)
    num_dias = total_hours // 24 if total_hours else 0

    # --------------------------------------------------
    # 2) Dimensões do modelo
    # --------------------------------------------------
    num_vars = num_periods
    num_restricoes = sum(
        1 for r in restrictions if selected_restrictions.get(r["Key"], False)
    )

    rhs_values = need.copy()

    # --------------------------------------------------
    # 3) Construção da matriz de restrições inicial
    # --------------------------------------------------
    initial_constraints_coefficients = np.zeros(
        (num_periods, num_periods), dtype=int
    )

    initial_constraints_coefficients = preencher_restricoes(
        initial_constraints_coefficients,
        restrictions,
        selected_restrictions,
        num_periods,
        need,
    )

    # --------------------------------------------------
    # 4) Cache da matriz
    # --------------------------------------------------
    constraints_coefficients = load_data(cache_path)

    if (
        constraints_coefficients is None
        or not isinstance(constraints_coefficients, np.ndarray)
        or constraints_coefficients.size == 0
        or update_cache
    ):
        constraints_coefficients = initial_constraints_coefficients.copy()
        save_data(constraints_coefficients, cache_path)

    # --------------------------------------------------
    # 5) Métricas iniciais
    # --------------------------------------------------
    initial_density = (
        calculate_density(initial_constraints_coefficients)
        if num_periods > 0
        else None
    )

    # --------------------------------------------------
    # 6) Retorno estruturado
    # --------------------------------------------------
    return {
        "need": need,
        "num_vars": num_vars,
        "num_restricoes": num_restricoes,
        "rhs_values": rhs_values,
        "num_periods": num_periods,
        "num_dias": num_dias,
        "initial_constraints_coefficients": initial_constraints_coefficients,
        "constraints_coefficients": constraints_coefficients,
        "initial_density": initial_density,
    }

# Interface do Streamlit
st.title(t("sim_main_title"))

# Inicializa a variável default_need vazia
default_need = []
need_input = None
num_periods = None

initial_constraints_coefficients = []

# Carrega os dados do arquivo, se existirem
initial_constraints_coefficients = load_data('initial_constraints_coefficients.json')

st.subheader(t("lbl_profile_section"))

profile_key = st.selectbox(
    t("lbl_select_profile"),
    options=["Custom"] + list(get_profiles().keys()),
    format_func=lambda k: t("prof_custom") if k == "Custom"
    else f"{k} — {get_profiles()[k]['description']}"
)

# Call get_restrictions() when needed
restrictions = get_restrictions()
PROFILES = get_profiles() # Local alias for compatibility logic if needed

#-----------------------------------
# Profile Management | Restrictions
# ----------------------------------
if "last_profile" not in st.session_state:
    st.session_state.last_profile = None

profile_changed = profile_key != st.session_state.last_profile

ALL_RESTRICTIONS = [
    "cobertura_necessidade",
    "limite_diario",
    "pausa_45_minutos",
    "divisao_pausa1530",
    "divisao_pausa3015",
    "repouso_diario_minimo",
    "repouso_diario_reduzido",
    "limite_semanal",
    "limite_quinzenal",
    "repouso_semanal",
    "repouso_quinzenal",
    "descanso_apos_trabalho",
]

profile_restrictions = {k: False for k in ALL_RESTRICTIONS}

if profile_key != "Custom":
    profile = PROFILES[profile_key]
    if profile["restrictions_on"] == "ALL":
        for k in profile_restrictions:
            profile_restrictions[k] = True
    else:
        for k in profile["restrictions_on"]:
            profile_restrictions[k] = True

# 🔥 APLICAR PROFILE AO STATE (ANTES DOS WIDGETS)
if profile_key != "Custom" and profile_changed:
    for k, v in profile_restrictions.items():
        st.session_state[k] = v

    # 🔥 aplicar parâmetros do profile
    for param_key, param_value in profile.get("params", {}).items():
        st.session_state[param_key] = param_value

    # cobertura sempre ligada
    st.session_state["cobertura_necessidade"] = True

    # sincronizar rádio de pausas
    if profile_restrictions.get("pausa_45_minutos"):
        st.session_state["restricoes_pausas"] = "45 minutes"
    elif profile_restrictions.get("divisao_pausa1530"):
        st.session_state["restricoes_pausas"] = "15+30 split"
    elif profile_restrictions.get("divisao_pausa3015"):
        st.session_state["restricoes_pausas"] = "30+15 split"
    else:
        st.session_state["restricoes_pausas"] = "None"

    st.session_state.last_profile = profile_key    

#-----------------------------------
# Parametros Globais | Restrictions
# ----------------------------------
with st.expander(t("global_params"), expanded=True):
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,4])
    with col1:
        st.write(t("col_global"))
        
        total_hours = st.number_input(t("lbl_horizon"), min_value=1, key="horizon_hours", value=24)
        period_minutes = st.number_input(t("lbl_slot"), min_value=1, key="slot", value=15)
        tolerance_demands = st.number_input(t("lbl_tolerance"), min_value=0.01, key="tolerance_coverage")

    with col2:
        st.write(t("col_algo"))

        variable_type = st.selectbox(
            t("lbl_variable"), 
            ["Integer", "Binary", "Continuous"], 
            format_func=lambda x: t({"Integer": "opt_integer", "Binary": "opt_binary", "Continuous": "opt_continuous"}.get(x, x)),
            key="variable_type"
        )
        solver_param_type = st.selectbox(t("lbl_solver"), ["SCIP", "GLOP"], key="solver")
        acceptable_percentage = st.number_input(t("lbl_density"), min_value=0.01, key="acceptable_density")
        enable_symmetry_breaking = st.checkbox(t("lbl_symmetry"), value=False)
        max_time_seconds = st.number_input(t("lbl_time"), min_value=10, value=300, step=10, help="Max solver runtime per instance")

    with col3:
        st.write(t("col_iter"))

        limit_iteration = st.number_input(t("lbl_limit_iter"), min_value=0, key="limit_iteration")
        limit_level_relaxation = st.number_input(t("lbl_relax"), min_value=0, key="relaxation")
        
    with col4:
        st.write(t("col_drivers"))
        penalty = st.number_input(t("lbl_penalty"), min_value=0.01, key="penalty_unmet")
        cap_tasks_per_driver_per_slot = st.number_input(t("lbl_cap"), min_value=1, key="cap_tasks_per_driver_per_slot") #cap_tasks_per_driver_per_slot
        limit_workers = st.number_input(t("lbl_drivers"), min_value=1, key="limit_workers", value=100)
        
    with col5:
        fixar_valores = st.checkbox(t("lbl_set_val"))
        num_periods = (total_hours * 60) // period_minutes

        # 1) inicialização única
        if "default_need" not in st.session_state:
            st.session_state.default_need = np.random.randint(1, 11, size=num_periods).tolist()

        if "need_text" not in st.session_state:
            st.session_state.need_text = ", ".join(map(str, st.session_state.default_need))

        if "last_fixar_state" not in st.session_state:
            st.session_state.last_fixar_state = fixar_valores

        # 2) só reage se o checkbox MUDAR de estado
        if fixar_valores != st.session_state.last_fixar_state:
            if fixar_valores:
                # congelar valores (NAO FAZER NADA - PRESERSVAR O QUE O USUARIO DIGITOU)
                # O input atual já está em st.session_state.need_text
                pass 
                # st.session_state.need_text = ", ".join(map(str, st.session_state.default_need))
            else:
                # gerar novos valores apenas UMA vez
                st.session_state.default_need = np.random.randint(1, 11, size=num_periods).tolist()
                st.session_state.need_text = ", ".join(map(str, st.session_state.default_need))

            st.session_state.last_fixar_state = fixar_valores

        # 3) widget
        need_input = st.text_area(
            t("lbl_slot_demand"),
            key="need_text",
            height=210
        )

        need = [x.strip() for x in need_input.split(",") if x.strip()]
        st.write(f"{t('lbl_total_demand')} {len(need)}")

        # parsing seguro
        needNew = []
        for x in need:
            try:
                needNew.append(float(x))
            except ValueError:
                pass
    
#-----------------------------------
# Objective Function | Restrictions
# ----------------------------------
#-----------------------------------
# Objective Function | Restrictions
# ----------------------------------
st.subheader(t("lbl_obj"))
with st.expander(t("lbl_obj"), expanded=True):
        # GET DYNAMIC DICTS
        lex_objectives = get_lex_objectives()
        objective_details = get_objective_details()
        
        radio_selection_object = st.radio(
                t("lbl_obj_strat"),
                options=list(lex_objectives.keys()),
                key="funcao_Objetivo"
            )
        with st.expander(t("lbl_breakdown"), expanded=False):
            st.markdown(f"### {t('lbl_breakdown')}")
            rows = objective_details[radio_selection_object]
            render_math_table(rows)
            st.markdown(f"### {t('lbl_lex_struct')}")
            for obj in lex_objectives[radio_selection_object]:
                st.markdown(f"**Priority {obj['priority']} — {obj['name']}**")
                st.latex(obj["formula"])
                st.caption(obj["description"])

with st.expander(t("lbl_restr"), expanded=True):
    selected_restrictions = {}
    col1, col2, col3 = st.columns([2, 5, 5])
    # Radiobuttons na primeira coluna
    with col1:
        # st.write("Break Options")
        radio_selection = st.radio(
            t("lbl_break_opt"), 
            options=["None", "45 minutes", "15+30 split", "30+15 split"],
            format_func=lambda x: t({
                "None": "opt_none", 
                "45 minutes": "opt_45_min", 
                "15+30 split": "opt_15_30", 
                "30+15 split": "opt_30_15"
            }.get(x, x)),
            index=1,  # Inicialmente nenhuma pausa selecionada
            key="restricoes_pausas"
        )
        
        for restriction in restrictions:  # Primeira metade das restrições
            checkbox_label = f"{restriction['Description']} | {restriction['Formula']}"
        
            # Verificando se a restrição está relacionada a pausas
            if restriction["Key"] == "pausa_45_minutos":
                # Se a opção de "Pausa 45 minutos" foi selecionada no radio button
                selected_restrictions[restriction["Key"]] = (radio_selection == "45 minutes")
            elif restriction["Key"] == "divisao_pausa1530":
                # Se a opção "Divisão Pausa 15:30" foi selecionada no radio button
                selected_restrictions[restriction["Key"]] = (radio_selection == "15+30 split")
            elif restriction["Key"] == "divisao_pausa3015":
                # Se a opção "Divisão Pausa 30:15" foi selecionada no radio button
                selected_restrictions[restriction["Key"]] = (radio_selection == "30+15 split")
    # Dividir as restrições entre as outras colunas
    with col2:
        for restriction in restrictions:  # Primeira metade das restrições
            checkbox_label = f"{restriction['Description']} | {restriction['Formula']}"
            
            if restriction["Key"] == "cobertura_necessidade":
                default_checked = restriction["Key"] == "cobertura_necessidade"
                selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])
            elif restriction["Key"] == "limite_diario":
                default_checked = restriction["Key"] == "limite_diario"
                selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])
            elif restriction["Key"] == "repouso_diario_minimo":
                default_checked = restriction["Key"] == "repouso_diario_minimo"
                selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])
            elif restriction["Key"] == "repouso_diario_reduzido":
                default_checked = restriction["Key"] == "repouso_diario_reduzido"
                selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])
    with col3:
        for restriction in restrictions:  # Segunda metade das restrições
            checkbox_label = f"{restriction['Description']} | {restriction['Formula']}"
            
            # Verificando se a restrição está relacionada a pausas
            if (restriction["Key"] != "pausa_45_minutos" 
                and restriction["Key"] != "divisao_pausa1530"
                and restriction["Key"] != "divisao_pausa3015"
                and restriction["Key"] != "cobertura_necessidade"
                and restriction["Key"] != "limite_diario"
                and restriction["Key"] != "repouso_diario_minimo" 
                and restriction["Key"] != "repouso_diario_reduzido"):
                selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])

st.subheader(t("lbl_cache"))
with st.expander(t("lbl_params"), expanded=True):
    atualizarFicheiro = st.checkbox(t("lbl_update_file"), value=False, help="If enabled, the options below will be disabled.")
    if atualizarFicheiro:
        constraints_coefficients = initial_constraints_coefficients
        save_data(initial_constraints_coefficients, 'initial_constraints_coefficients.json')
        save_data(constraints_coefficients, 'constraints_coefficients.json')

inputs = prepare_optimization_inputs(
    need_input=need_input,
    total_hours=total_hours,
    period_minutes=period_minutes,
    restrictions=get_restrictions(),
    selected_restrictions=selected_restrictions,
    update_cache=atualizarFicheiro,
)

need = inputs["need"]
num_vars = inputs["num_vars"]
num_restricoes = inputs["num_restricoes"]
rhs_values = inputs["rhs_values"]
num_periods = inputs["num_periods"]
num_dias = inputs["num_dias"]
initial_constraints_coefficients = inputs["initial_constraints_coefficients"]
constraints_coefficients = inputs["constraints_coefficients"]
initial_density_matrix = inputs["initial_density"]

st.subheader(t("lbl_ml_mode"))
with st.expander(t("lbl_ml_train"), expanded=False):
    st.markdown(f"### {t('lbl_ml_desc')}")
    st.caption(f"""
    {t('lbl_f1_desc')}
    {t('lbl_f2_desc')}
    """)
    
    train_clicked = st.button(t("btn_train_ml"), key="train_ml")
    if train_clicked:
        st.session_state["run_train_ml"] = True

if st.session_state.run_train_ml:
    st.info("Starting model training... this may take a few minutes.")
    progress = st.progress(0)
    log = st.empty()

    progress.progress(0.3)
    log.write(t("ml_training_progress")) # "Training f₁ and f₂ models..."

    result = run_ml_training()

    progress.progress(1.0)
    log.write(t("ml_training_done")) # "Training completed!"

    st.session_state.train_ml_result = result
    st.session_state.run_train_ml = False    
    
if st.session_state.train_ml_result:

    if st.session_state.train_ml_result["success"]:
        st.success(t("ml_training_done")) # "Training process completed!"
        render_ml_training_results(
            st.session_state.train_ml_result["results"]
        )
    else:
        st.error(t("ml_training_error")) # "Error during training:"
        st.code(st.session_state.train_ml_result["error"])
            
with st.expander(t("lbl_ml_dataset"), expanded=False):
    
    st.markdown(f"### {t('lbl_gen_dataset')}")

    col_ml_1, col_ml_2, col_ml_3 = st.columns(3)

    with col_ml_1:
        num_instances = st.number_input(
            t("lbl_num_instances"), # "Number of instances"
            min_value=1,
            max_value=500,
            value=10,
            step=1,
        )
        num_periods_ml = st.number_input(
            t("lbl_periods_per_inst"), # "Periods per instance"
            min_value=4,
            max_value=96,
            value=24,
            step=4,
        )

    with col_ml_2:
        limit_workers_ml = st.number_input(
            t("lbl_drivers_limit"), # "Drivers (limit)"
            min_value=1,
            max_value=200,
            value=10,
            step=1,
        )
        max_demands_per_driver_ml = st.number_input(
            t("lbl_max_periods_driver"), # "Maximum number of periods per driver"
            min_value=1,
            max_value=500,
            value=20,
            step=1,
        )

    with col_ml_3:
        demand_min = st.number_input(
            t("lbl_min_demand"), # "Minimum demand per period"
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )
        demand_max = st.number_input(
            t("lbl_max_demand"), # "Maximum demand per period"
            min_value=1,
            max_value=100,
            value=5,
            step=1,
        )
        random_seed = st.number_input(
            t("lbl_seed"), # "Random seed"
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

    st.caption(
        t("lbl_dataset_note") # "Note: Small instances..."
    )

    dataset_clicked = st.button(t("btn_gen_dataset"), key="dataset_ml")
    if dataset_clicked:
        st.session_state["run_dataset_ml"] = True

if st.session_state.run_dataset_ml:

    rng = np.random.default_rng(random_seed)

    f1_rows_all: list[dict] = []
    f2_rows_all: list[dict] = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for inst in range(int(num_instances)):
        progress = (inst + 1) / num_instances
        progress_bar.progress(progress)
        status_text.text(f"Gerando instância {inst + 1} de {int(num_instances)}...")

        # ----------------------------------------------
        # 1) Gerar demanda sintética para esta instância
        # ----------------------------------------------
        need_ml = rng.integers(
            low=demand_min,
            high=demand_max + 1,
            size=int(num_periods_ml),
        ).tolist()

        # ----------------------------------------------
        # 2) Solução heurística (greedy)
        # ----------------------------------------------
        greedy_alloc = greedy_initial_allocation(
            need=need_ml,
            limit_workers=int(limit_workers_ml),
            max_demands_per_driver=int(max_demands_per_driver_ml),
            assignment_scorer_fn=None,  # pode ligar ML depois
        )
        total_workers_greedy = int(greedy_alloc.sum())
        
        try:
            (
                solver,
                status,
                total_active_drivers_opt,
                total_assigned_slots_opt,
                workers_schedule_opt,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                matrix_allocation_opt,
                _,
                _,
            ) = run_solver_with_mode(
                mode="Exact",
                need=need_ml,
                variable_type=variable_type,
                constraints_coefficients=constraints_coefficients,
                selected_restrictions=selected_restrictions,
                solver_param_type=solver_param_type,
                densidade_aceitavel=acceptable_percentage,
                limit_workers=int(limit_workers_ml),
                limit_iteration=limit_iteration,
                limit_level_relaxation=limit_level_relaxation,
                max_demands_per_driver=int(max_demands_per_driver_ml),
                tolerance_demands=tolerance_demands,
                penalty=penalty,
                radio_selection_object=radio_selection_object,
            )

        except Exception as e:
            st.warning(
                f"Instância {inst}: erro ao resolver (Exact) → {e}"
            )
            continue

        if matrix_allocation_opt is None:
            st.warning(
                f"Instância {inst}: solver não retornou matriz válida."
            )
            continue                

        # --------------------------------------------------
        # 4) Normalização CANÔNICA (🔥 essencial 🔥)
        # --------------------------------------------------
        workers_schedule_opt, matrix_allocation_opt = (
            normalize_solver_outputs(
                demanda=need_ml,
                workers_schedule=workers_schedule_opt,
                matrix_allocation=matrix_allocation_opt,
            )
        )

        if workers_schedule_opt is None or matrix_allocation_opt is None:
            st.warning(
                f"Instância {inst}: normalização falhou. Ignorada."
            )
            continue

        # --------------------------------------------------
        # 5) Construção dos datasets f₁ e f₂ (NOVO builder)
        # --------------------------------------------------
        f1_rows = build_f1_rows(
            need=np.array(need_ml),
            workers_schedule=workers_schedule_opt,
            matrix_allocation=matrix_allocation_opt,
        )

        f2_rows = build_f2_rows(
            need=np.array(need_ml),
            workers_schedule=workers_schedule_opt,
            matrix_allocation=matrix_allocation_opt,
            kpis={
                "global_coverage": (
                    np.sum(
                        np.minimum(
                            workers_schedule_opt, need_ml
                        )
                    )
                    / max(1, sum(need_ml))
                ),
                "worker_efficiency": (
                    np.sum(
                        np.minimum(
                            workers_schedule_opt, need_ml
                        )
                    )
                    / max(1, sum(workers_schedule_opt))
                ),
                "operational_risk": (
                    np.sum(
                        np.maximum(
                            np.array(need_ml)
                            - np.array(workers_schedule_opt),
                            0,
                        )
                    )
                    / max(1, sum(need_ml))
                ),
                "temporal_stability": 1.0,
            },
        )

        f1_rows_all.extend(f1_rows)
        f2_rows_all.extend(f2_rows)

        time.sleep(0.03)                

        # --------------------------------------------------
        # 6) Persistência em CSV
        # --------------------------------------------------
        f1_path, f2_path = save_datasets(
            f1_rows_all, f2_rows_all
        )

        progress_bar.progress(1.0)
        status_text.text(t("ml_gen_done"))

        st.success(
            t("ml_gen_success_detail").format(
                len(f1_rows_all), os.path.abspath(f1_path),
                len(f2_rows_all), os.path.abspath(f2_path)
            )
        )                

        st.info(f"{t('ml_dataset_folder')}\n{os.path.abspath(os.path.dirname(f1_path))}")

        if f1_rows_all:
            st.write(t("lbl_preview_f1"))
            st.dataframe(
                pd.DataFrame(f1_rows_all).head(),
                column_config={
                    "period": t("col_period"),
                    "worker": t("col_worker"),
                    "need": t("col_need"),
                    "allocated_workers": t("col_allocated"),
                    "worker_total_load": t("col_worker_load"),
                    "demand_gap": t("col_gap"),
                    "label": t("col_label")
                }
            )

        if f2_rows_all:
            st.write(t("lbl_preview_f2"))
            st.dataframe(
                pd.DataFrame(f2_rows_all).head(),
                column_config={
                    "num_periods": t("col_period"), # Reuse period or make specific
                    "total_demand": t("col_total_demand"),
                    "total_allocated": t("col_total_alloc"),
                    "uncovered_demand": t("col_uncovered"),
                    "avg_load": t("col_avg_load"),
                    "load_variance": t("col_var_load"),
                    "global_coverage": t("col_global_cov"),
                    "worker_efficiency": t("col_efficiency"),
                    "operational_risk": t("col_risk"),
                    "temporal_stability": t("col_stability"),
                    "label": t("col_label")
                }
            )

    st.session_state.run_dataset_ml = False    


st.subheader(t('lbl_opt_mode'))
with st.expander(t('lbl_params'), expanded=True):
    optimization_mode = st.radio(
        t('lbl_mode_select'),
        ["Exact", "Heuristic", "LNS"],
        format_func=lambda x: t({"Exact": "opt_exact", "Heuristic": "opt_heuristic", "LNS": "opt_lns"}.get(x, x)),
        index=0,
        horizontal=True
    )
    
    # LNS Parameters
    max_lns_iterations = 5
    if optimization_mode == "LNS":
        max_lns_iterations = st.slider(
            t('lbl_lns_iter'),
            min_value=1,
            max_value=20,
            value=5,
            help="Number of improvement iterations for LNS (more = better solution but slower)"
        )
        
    enable_assignment_maximization = st.checkbox(
        t('lbl_max_assign'),
        value=False,
        help="If enabled, the solver will try to assign as many slots as possible to active drivers (Good for Operations). If disabled, it minimizes assignments (Good for Efficiency/Academic)."
    )

st.subheader(t('lbl_elem_ops'))
with st.expander(t('lbl_params'), expanded=True):
    selected_operations = {}   
    for op in get_elemental_operations():
        selected_operations[op['Key']] = st.checkbox(
            f"{op['Description']} | {op['Formula']}",
            key=f"op_{op['Key']}",
            help="Enable this elementary operation"
        )

    # =========================
    # UI CONDICIONAL
    # =========================

    if st.session_state.get("op_troca_equacoes"):
        st.markdown(f"### 🔁 {t('op_swap')}")
        swap_row_1 = st.number_input(t('lbl_row_1'), min_value=0, key="swap_row_1")
        swap_row_2 = st.number_input(t('lbl_row_2'), min_value=0, key="swap_row_2")

    if st.session_state.get("op_multiplicacao_por_constante"):
        st.markdown(f"### ✖ {t('op_mult')}")
        mult_row = st.number_input(t('lbl_row'), min_value=0, key="mult_row")
        mult_const = st.number_input(t('lbl_const'), value=1, key="mult_const")

    if st.session_state.get("op_soma_multiplo_equacao"):
        st.markdown(f"### ➕ {t('op_sum')}")
        sum_row_base = st.number_input(t('lbl_base'), min_value=0, key="sum_row_base")
        sum_row_target = st.number_input(t('lbl_target'), min_value=0, key="sum_row_target")
        sum_multiplier = st.number_input(t('lbl_mult'), value=1, key="sum_multiplier")
        
    if st.session_state.get("op_soma_multiplo_equacao_automatica"):
        st.markdown(f"### ➕ {t('op_sum')} (Auto)")
        sum_row_base_auto = st.number_input(t('lbl_base'), min_value=0, key="sum_row_base_auto")
        sum_row_target_auto = st.number_input(t('lbl_target'), min_value=0, key="sum_row_target_auto")
        sum_multiplier_auto = st.number_input(t('lbl_mult'), value=1, key="sum_multiplier_auto")            

                    
# Buttons Layout
c_run_1, c_run_2 = st.columns([1,1])
btn_run = c_run_1.button(t("run_opt"), key="btn_run_optimization", use_container_width=True)
btn_bench = c_run_2.button(t("btn_bench"), key="btn_bench", use_container_width=True, help="Executes Exact, Heuristic and LNS sequentially and compares results.")

if btn_bench:
    st.markdown(f"### {t('header_bench')}")
    
    bench_results = []
    modes_to_test = ["Heuristic", "LNS", "Exact"] # Heuristic first (fastest)
    
    # Parse inputs (robust)
    need_raw = [x.strip() for x in need_input.split(",") if x.strip()]
    need = []
    for x in need_raw:
        try:
            need.append(int(float(x)))
        except ValueError:
            pass # ignore invalid

    
    if len(need) != num_periods:
        st.error(f"Input mismatch: {len(need)} slots vs {num_periods} expected.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        lns_history_data = [] # To capture Pareto Front
        
        for i, mode in enumerate(modes_to_test):
            status_text.text(f"Running {mode}...")
            
            # Constraints
            constraints_coefficients = load_data('constraints_coefficients.json')
            
            # Timer
            start_time = time.time()
            
            # Run Solver
            try:
                (solver, status, total_active, total_slots, ws, ts, _, _, _, _, _, _, iterations_data_result, alloc, logs) = run_solver_with_mode(
                    mode, need, variable_type, constraints_coefficients, selected_restrictions,
                    solver_param_type, acceptable_percentage, limit_workers, limit_iteration,
                    limit_level_relaxation, cap_tasks_per_driver_per_slot, tolerance_demands, penalty,
                    swap_rows=None, multiply_row=None, add_multiple_rows=None,
                    radio_selection_object="Minimize Total Number of Drivers", # FORCE MINIMIZATION FOR BENCHMARK
                    enable_symmetry_breaking=enable_symmetry_breaking,
                    time_limit_seconds=max_time_seconds,
                    max_lns_iterations=max_lns_iterations,
                    enable_assignment_maximization=enable_assignment_maximization
                )
                
                if mode == "LNS":
                    lns_history_data = iterations_data_result
                
                elapsed = time.time() - start_time
                
                # Calculate KPIs
                # compute_kpis definition: compute_kpis(demanda, workers_schedule=None, matrix_allocation=None)
                kpis, coverage_val, safety_val = compute_kpis(need, workers_schedule=ws, matrix_allocation=alloc)
                
                coverage_pct = (kpis.get("global_coverage") or 0) * 100
                efficiency_pct = (kpis.get("worker_efficiency") or 0) * 100
                idle_pct = 100 - efficiency_pct
                
                sol_status_str = get_solver_status_description(status) if isinstance(status, int) else str(status)
                
                # SHIFT_MIN check
                illegal_shifts = check_shift_violations(alloc, min_shift_periods=16)

                bench_results.append({
                    "Method": mode,
                    "Sol. Status": sol_status_str,
                    "Time (s)": round(elapsed, 2),
                    "Active Drivers": total_active,
                    "Assigned Slots": total_slots,
                    "Coverage (%)": round(coverage_pct, 2),
                    "Illegal Shifts": illegal_shifts,
                    "Efficiency (%)": round(efficiency_pct, 2)
                })
                
            except Exception as e:
                 st.error(f"Error in {mode}: {e}")
                 bench_results.append({"Method": mode, "Sol. Status": "ERROR", "Time (s)": 0})
                 import traceback
                 traceback.print_exc()
            
            progress_bar.progress((i + 1) / len(modes_to_test))
            
        status_text.text("Benchmark Completed!")
        progress_bar.empty()
        
        # Display Table
        st.markdown("#### 📊 Results Table")
        df_bench = pd.DataFrame(bench_results)
        
        # Highlight best values
        st.dataframe(df_bench.style.highlight_max(subset=["Coverage (%)", "Efficiency (%)"], color='lightgreen')
                                   .highlight_min(subset=["Time (s)", "Active Drivers", "Illegal Shifts"], color='lightgreen'))
        
        # ------------------------------------------------------------------
        # PARETO FRONT VISUALIZATION
        # ------------------------------------------------------------------
        if lns_history_data:
            st.divider()
            st.markdown("#### 📉 LNS Exploration: Coverage vs. Resource Trade-off")
            st.markdown("This chart visualizes the search space explored by the LNS solver, showing different 'Pareto-optimal' candidates.")
            
            total_demand = sum(need)
            h_rows = []
            for h in lns_history_data:
                cov_pct = (h['total_assigned_slots'] / total_demand * 100) if total_demand > 0 else 0
                h_rows.append({
                    "Iteration": int(h['iteration']),
                    "Active Drivers": float(h['total_active_drivers']) if h['total_active_drivers'] is not None else 0.0,
                    "Coverage (%)": float(cov_pct),
                    "State": "INITIAL" if h['status'] == "INITIAL" else ("IMPROVED" if h.get("improved") else "REJECTED")
                })
            
            df_history = pd.DataFrame(h_rows)
            # Ensure proper types for plotting
            df_history["Active Drivers"] = pd.to_numeric(df_history["Active Drivers"], errors='coerce')
            df_history["Coverage (%)"] = pd.to_numeric(df_history["Coverage (%)"], errors='coerce')
            
            # Filter out invalid or zero-driver points
            df_history = df_history[df_history['Active Drivers'] > 0].dropna()
            
            if not df_history.empty:
                # Add jitter for visualization of overlapping points
                df_jitter = df_history.copy()
                # Jitter proportional to scale: ~2% for drivers, ~1% for coverage
                df_jitter["Active Drivers"] += np.random.normal(0, 1.5, len(df_jitter)) 
                df_jitter["Coverage (%)"] += np.random.normal(0, 0.5, len(df_jitter))

                # Highlight the iterations with fewer drivers using Plotly for better tooltips
                c1, c2 = st.columns([1, 1])
                with c1:
                    import plotly.express as px
                    fig = px.scatter(
                        df_jitter, 
                        x="Active Drivers", 
                        y="Coverage (%)", 
                        color="State",
                        symbol="State",
                        hover_data=["Iteration", "State"],
                        title="LNS Exploration History",
                        color_discrete_map={"INITIAL": "#31333F", "IMPROVED": "#28a745", "REJECTED": "#dc3545"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("*(Note: Jitter added to expose overlapping iterations. Hover to see details)*")
                
                with c2:
                    st.write("**💡 Scientific Insight:**")
                    st.write(f"Total iterations recorded: **{len(df_history)}**")
                    st.write("Each dot represents a schedule attempt. The 'REJECTED' dots around the 'INITIAL' point show that LNS is trying to optimize but being blocked by strict coverage requirements.")
                    
                    # Show top alternative candidate
                    df_filtered = df_history[df_history['Coverage (%)'] > 90]
                    if df_filtered.empty and not df_history.empty:
                        max_cov = df_history['Coverage (%)'].max()
                        df_filtered = df_history[df_history['Coverage (%)'] >= max_cov * 0.95]
                    
                    if not df_filtered.empty:
                        alt_cand = df_filtered.sort_values(by="Active Drivers").iloc[0]
                        st.info(f"**Efficient Alternative Found:** {alt_cand['Active Drivers']} drivers ({alt_cand['Coverage (%)']:.1f}% coverage)")
                    else:
                        st.warning("*(No improvements found yet: Heuristic solution remains the best detected local point)*")
                
                # Show full history table for transparency
                with st.expander("📝 View Detailed LNS Iteration Log"):
                    st.dataframe(df_history.sort_values(by="Iteration"), use_container_width=True)
            else:
                st.write("*(No valid LNS iterations found to plot)*")

        # Download CSV
        csv = df_bench.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results CSV", csv, "benchmark_results.csv", "text/csv")

        # ------------------------------------------------------------------
        # AUTO INTERPRETER
        # ------------------------------------------------------------------
        st.divider()
        st.markdown("### 🧠 Automated Academic Interpretation")
        
        res_dict = {row['Method']: row for row in bench_results if row.get('Sol. Status') != 'ERROR'}
        
        if "LNS" in res_dict and "Heuristic" in res_dict and "Exact" in res_dict:
            lns = res_dict["LNS"]
            heur = res_dict["Heuristic"]
            exact = res_dict["Exact"]
            
            # --- 1. Trade-off ---
            st.markdown("#### 1. Efficiency vs Time (The Grand Trade-off)")
            
            time_gain_vs_exact = (exact["Time (s)"] / lns["Time (s)"]) if lns["Time (s)"] > 0 else 0
            
            if time_gain_vs_exact > 1.0:
                 st.write(f"The **LNS** method demonstrated a significant speed advantage, being **{time_gain_vs_exact:.1f}x faster** than the Exact method.")
            else:
                 st.write(f"The LNS method had comparable timing to the Exact method ({time_gain_vs_exact:.1f}x).")
                 
            if lns["Active Drivers"] <= exact["Active Drivers"]:
                st.write(f"Remarkably, it achieved this speed while matching (or beating) the Exact solution quality ({lns['Active Drivers']} drivers).")
            else:
                 diff = lns["Active Drivers"] - exact["Active Drivers"]
                 if exact["Active Drivers"] > 0:
                     diff_pct = (diff / exact["Active Drivers"]) * 100
                     st.write(f"It came very close to the optimal solution, using only **{diff} more drivers ({diff_pct:.1f}%)** than the ground truth.")
                 else:
                      st.write(f"It utilized {diff} drivers (Exact method returned 0 drivers).")

            # --- 2. Key Metrics ---
            st.markdown("#### 2. Key Metrics Analysis")
            
            drivers_saved = heur["Active Drivers"] - lns["Active Drivers"]
            drivers_saved_pct = (drivers_saved / heur["Active Drivers"] * 100) if heur["Active Drivers"] > 0 else 0
            
            eff_gain = lns["Efficiency (%)"] - heur["Efficiency (%)"]
            
            st.write(f"• **Active Drivers (Cost)**: LNS reduced the fleet by **{drivers_saved_pct:.1f}%** compared to the Heuristic baseline (saving {drivers_saved} drivers).")
            st.write(f"• **Efficiency**: LNS improved workforce utilization by **+{eff_gain:.1f} pp**, reducing idle time.")
            
            # --- 3. Operational Rigor ---
            st.markdown("#### 3. Operational Rigor (The 'Hidden' Truth)")
            if heur["Illegal Shifts"] > exact["Illegal Shifts"]:
                ratio = heur["Illegal Shifts"] / exact["Illegal Shifts"] if exact["Illegal Shifts"] > 0 else heur["Illegal Shifts"]
                st.write(f"⚠️ **Note on Operational Quality**: The Heuristic generated **{heur['Illegal Shifts']} illegal shifts**.")
                st.write(f"The Exact solver was **{ratio:.1f}x cleaner**, with only {exact['Illegal Shifts']} violations.")
                st.success("Mathematical optimization naturally 'bundles' tasks to minimize drivers, which leads to more legal and stable shifts than rules-based heuristics.")
                if exact["Illegal Shifts"] > 0:
                    st.info("💡 *Tip: To reach 0 illegal shifts, enable 'Duração Mínima de Turno' in the configuration panel.*")
            else:
                st.write("Both solvers maintained high operational rigor with minimal illegal shifts.")

            # --- 4. Summary ---
            st.markdown("#### 4. Summary for Paper")
            
            # Defensive checks for division by zero
            exact_drivers = exact.get('Active Drivers', 0) or 1  # Avoid division by zero
            lns_drivers = lns.get('Active Drivers', 0) or 0
            lns_vs_exact_pct = (lns_drivers / exact_drivers * 100) if exact_drivers > 0 else 0
            time_pct = (1 / time_gain_vs_exact * 100) if time_gain_vs_exact > 0 else 0
            
            summary_text = (
                f"The computational results demonstrate that the proposed LNS algorithm successfully balances solution quality and execution time. "
                f"It outperformed the constructive heuristic by reducing active drivers by {drivers_saved_pct:.1f}% and improving efficiency by {eff_gain:.1f}pp. "
                f"Compared to the exact method, LNS achieved {lns_vs_exact_pct:.1f}% of the optimal quality "
                f"while requiring only {time_pct:.1f}% of the computational time."
            )
            st.info(summary_text)

        elif "LNS" not in res_dict:
            st.warning("LNS run failed or missing. Cannot generate full interpretation.")


if btn_run:
    
    # ⏱️ Timer & Progress
    ph_timer = st.empty()
    ph_progress = st.progress(0)
    start_time_opt = time.time()
    ph_progress.progress(5)

    # Inicializar variáveis para as operações
    swap_rows = None
    multiply_row = None
    add_multiple_rows = None
    iterations_data_result = []

    with st.spinner(t("loading_opt")):
        try:
            # Sempre cria as colunas ANTES de validar need
            col_resultsItIOpI, col_resultsItIOpII, col_resultsItIOpIII = st.columns(3)
            col_resultsI_col = st.columns(1)[0]
            need = list(map(int, need_input.split(',')))

            # Exibir a matriz de restrições antes da otimização
            if len(need) != num_periods:
                st.error(f"The input must have exactly {num_periods} values ​​(1 for each period of {period_minutes} minutes).")
            else:
                with col_resultsI_col:
                    swap_rows_c = selected_operations["troca_equacoes"] #st.checkbox("Troca de Equações")
                    multiply_row_c = selected_operations["multiplicacao_por_constante"] #st.checkbox("Multiplicação por Constante")
                    add_multiple_rows_c = selected_operations["soma_multiplo_equacao"] #st.checkbox("Somar Múltiplo de uma Equação a Outra")
                    add_multiple_rows_c_auto = selected_operations["soma_multiplo_equacao_automatica"] #st.checkbox("Somar Múltiplo de uma Equação a Outra - Automático")
                    
                    if (not swap_rows_c and not multiply_row_c and not add_multiple_rows_c and not add_multiple_rows_c_auto):
                            ph_progress.progress(20, text="Loading constraints...")
                            constraints_coefficients = load_data('constraints_coefficients.json')                        
                            ph_progress.progress(40, text="Running Solver (this may take time)...")                        
                            (solver, 
                             status, 
                             total_active_drivers, 
                             total_assigned_slots, 
                             workers_schedule, 
                             tasks_schedule, 
                             driving_hours_per_driver, 
                             constraints_coefficients, 
                             initial_density, 
                             final_density, 
                             statistics_result, 
                             msg, 
                             iterations_data_result, 
                             matrix_allocation, 
                             solver_logs
                            ) = run_solver_with_mode(
                            optimization_mode, 
                            need, 
                            variable_type, 
                            constraints_coefficients, 
                            selected_restrictions, 
                            solver_param_type, 
                            acceptable_percentage, 
                            limit_workers, 
                            limit_iteration, 
                            limit_level_relaxation, 
                            cap_tasks_per_driver_per_slot, 
                            tolerance_demands, 
                            penalty, 
                            swap_rows=None, 
                            multiply_row=None, 
                            add_multiple_rows=None, 
                            radio_selection_object=radio_selection_object,
                            enable_symmetry_breaking=enable_symmetry_breaking,
                            time_limit_seconds=max_time_seconds,
                            max_lns_iterations=max_lns_iterations,
                            enable_assignment_maximization=enable_assignment_maximization
                            )
                            # Pós-processamento Timer
                            elapsed_opt = time.time() - start_time_opt
                            ph_progress.progress(100, text="Completed!")
                            ph_timer.success(f"⏱️ Optimization finished in {elapsed_opt:.2f} seconds")
                    else:
                        if swap_rows_c:
                            with col_resultsItIOpI:
                                constraints_coefficients = load_data('constraints_coefficients.json') 
                                swap_row_1 = len(constraints_coefficients)-1
                                swap_row_2 = len(constraints_coefficients)-1
                                swap_rows = swap_row_1, swap_row_2

                                (solver, 
                                status, 
                                total_active_drivers,
                                total_assigned_slots,            
                                workers_schedule,
                                tasks_schedule,
                                driving_hours_per_driver, 
                                constraints_coefficients, 
                                initial_density, 
                                final_density, 
                                statistics_result, 
                                msg, 
                                iterations_data_result, 
                                matrix_allocation,
                                solver_logs
                                ) = run_solver_with_mode(
                                    optimization_mode,
                                    need,
                                    variable_type,
                                    constraints_coefficients,
                                    selected_restrictions,
                                    solver_param_type,
                                    acceptable_percentage,                            
                                    limit_workers,
                                    limit_iteration,
                                    limit_level_relaxation,
                                    cap_tasks_per_driver_per_slot,
                                    tolerance_demands,
                                    penalty,
                                    swap_rows=swap_rows, 
                                    multiply_row=None, 
                                    add_multiple_rows=None, 
                                    radio_selection_object=radio_selection_object,
                                    enable_symmetry_breaking=enable_symmetry_breaking,
                                    time_limit_seconds=max_time_seconds,
                                    max_lns_iterations=max_lns_iterations
                                )                            
                        
                        if multiply_row_c:
                            with col_resultsItIOpII:
                                constraints_coefficients = load_data('constraints_coefficients.json') 
                                mult_row = len(constraints_coefficients)-1
                                
                                if mult_const != 0:
                                    multiply_row = (mult_row, mult_const)
                                    constraints_coefficients = load_data('constraints_coefficients.json')
                            
                                    (solver, 
                                    status, 
                                    total_active_drivers, 
                                    total_assigned_slots,             
                                    workers_schedule,
                                    tasks_schedule,
                                    driving_hours_per_driver, 
                                    constraints_coefficients, 
                                    initial_density, 
                                    final_density, 
                                    statistics_result, 
                                    msg, 
                                    iterations_data_result, 
                                    matrix_allocation,
                                    solver_logs
                                    ) = run_solver_with_mode(
                                        optimization_mode,
                                        need,
                                        variable_type,
                                        constraints_coefficients,
                                        selected_restrictions,
                                        solver_param_type,
                                        acceptable_percentage,                            
                                        limit_workers,
                                        limit_iteration,
                                        limit_level_relaxation,
                                        cap_tasks_per_driver_per_slot,
                                        tolerance_demands,
                                        penalty,
                                        swap_rows=None, 
                                        multiply_row=multiply_row, 
                                        add_multiple_rows=None,
                                        radio_selection_object=radio_selection_object,
                                        enable_symmetry_breaking=enable_symmetry_breaking,
                                        time_limit_seconds=max_time_seconds,
                                        max_lns_iterations=max_lns_iterations
                                    )                            
                                else:
                                    st.warning("The multiplication constant cannot be zero!")
                        if add_multiple_rows_c:
                            with col_resultsItIOpIII:
                                
                                constraints_coefficients = load_data('constraints_coefficients.json')
                                sum_row_base = len(constraints_coefficients)-1, 
                                sum_row_target = len(constraints_coefficients)-1, 
                                
                                add_multiple_rows = sum_row_base, sum_row_target, sum_multiplier

                                (solver, 
                                status, 
                                total_active_drivers, # 3
                                total_assigned_slots, # 4            
                                workers_schedule,
                                tasks_schedule,
                                driving_hours_per_driver, 
                                constraints_coefficients, 
                                initial_density, 
                                final_density, 
                                statistics_result, 
                                msg, 
                                iterations_data_result, 
                                matrix_allocation,
                                solver_logs
                                ) = run_solver_with_mode(
                                    optimization_mode,
                                    need,
                                    variable_type,
                                    constraints_coefficients,
                                    selected_restrictions,
                                    solver_param_type,
                                    acceptable_percentage,                            
                                    limit_workers,
                                    limit_iteration,
                                    limit_level_relaxation,
                                    cap_tasks_per_driver_per_slot,
                                    tolerance_demands,
                                    penalty,
                                    swap_rows=None, 
                                    multiply_row=None, 
                                    add_multiple_rows=add_multiple_rows,
                                    radio_selection_object=radio_selection_object,
                                    enable_symmetry_breaking=enable_symmetry_breaking,
                                    time_limit_seconds=max_time_seconds,
                                    max_lns_iterations=max_lns_iterations
                                )                            
                            

                    if add_multiple_rows_c_auto:
                        
                        constraints_coefficientsNew = load_data('constraints_coefficients.json')
                        constraints_coefficients = load_data('constraints_coefficients.json')

                        #Percorrer a matriz com o multiplicador -1, todas as linhas considerando o row2 sempre um a menor que o row1 e submetendo ao modelo.
                        for idx, row in enumerate(constraints_coefficients):
                            row1 = idx+1
                            row2 = idx
                            multiple = -1
                            # add_multiple_rows = sum_row_base_auto, sum_row_target_auto, sum_multiplier_auto
                            add_multiple_rows = row1, row2, multiple
                            
                            (solver, 
                            status, 
                            total_active_drivers, # 3
                            total_assigned_slots, # 4            
                            workers_schedule,
                            tasks_schedule,
                            driving_hours_per_driver, 
                            constraints_coefficients, 
                            initial_density, 
                            final_density, 
                            statistics_result, 
                            msg, 
                            iterations_data_result, 
                            matrix_allocation,
                            solver_logs
                            ) = run_solver_with_mode(
                                optimization_mode,
                                need,
                                variable_type,
                                constraints_coefficients,
                                selected_restrictions,
                                solver_param_type,
                                acceptable_percentage,                            
                                limit_workers,
                                limit_iteration,
                                limit_level_relaxation,
                                cap_tasks_per_driver_per_slot,
                                tolerance_demands,
                                penalty,
                                swap_rows=None, 
                                multiply_row=None, 
                                add_multiple_rows=add_multiple_rows,
                                radio_selection_object=radio_selection_object,
                                enable_symmetry_breaking=enable_symmetry_breaking,
                                time_limit_seconds=max_time_seconds
                            )                                        

                            # Dentro da sua função solve_shift_schedule
                            final_density = calculate_density(constraints_coefficients)
                            
                            if final_density <= acceptable_percentage:
                                st.write(f"Final density {final_density} has reached acceptable limit ({acceptable_percentage}). Exiting loop.")
                                break

                # 🔧 Normalização para modos Heuristic/LNS
                print("DEBUG: POST-SOLVER - Starting normalization")
                if "workers_schedule" in locals() and "matrix_allocation" in locals():
                    if workers_schedule is None and matrix_allocation is not None:
                        try:
                            workers_schedule = list(np.sum(matrix_allocation, axis=1))
                        except Exception:
                            workers_schedule = []
                            
                print("DEBUG: POST-SOLVER - Normalization complete")
                
                print("DEBUG: POST-SOLVER - Checking msg")
                if msg is not None:
                    print("DEBUG: POST-SOLVER - msg is not None, parsing demanda")
                    # Converter a entrada de texto para uma lista de números
                    try:
                        demanda = list(map(int, need_input.split(',')))
                    except ValueError:
                        st.error("Por favor, insira os valores da demanda separados por vírgula e espaço.")
                        demanda = []
                
                print("DEBUG: POST-SOLVER - About to create Initial expander")        
                with st.expander("Initial", expanded=True):
                    print("DEBUG: POST-SOLVER - Inside Initial expander")
                    try:
                        import matplotlib.pyplot as plt
                        import io
                        
                        print("DEBUG: INITIAL - Creating all figures BEFORE rendering")
                        
                        # 🔒 Create ALL figures first (matplotlib only, no Streamlit)
                        with _MPL_LOCK:
                            print("DEBUG: INITIAL - Lock acquired for figure creation")
                            
                            # Chart A
                            print("DEBUG: INITIAL - Creating Chart A")
                            fig_A = plot_demand_vs_served(need, tasks_schedule)
                            buf_A = io.BytesIO()
                            fig_A.savefig(buf_A, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                            buf_A.seek(0)
                            plt.close(fig_A)
                            print("DEBUG: INITIAL - Chart A saved to buffer")
                            
                            # Chart B
                            print("DEBUG: INITIAL - Creating Chart B")
                            fig_B = plot_demand_gap(need, tasks_schedule)
                            buf_B = io.BytesIO()
                            fig_B.savefig(buf_B, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                            buf_B.seek(0)
                            plt.close(fig_B)
                            print("DEBUG: INITIAL - Chart B saved to buffer")
                            
                            # Chart C
                            print("DEBUG: INITIAL - Creating Chart C")
                            fig_C = plot_capacity_vs_utilization(
                                workers_schedule,
                                tasks_schedule,
                                cap_tasks_per_driver_per_slot
                            )
                            buf_C = io.BytesIO()
                            fig_C.savefig(buf_C, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                            buf_C.seek(0)
                            plt.close(fig_C)
                            print("DEBUG: INITIAL - Chart C saved to buffer")
                            
                            print("DEBUG: INITIAL - Lock released, all figures created")
                        
                        # 🖼️ Now display ALL images (no matplotlib, just Streamlit)
                        print("DEBUG: INITIAL - Creating columns for display")
                        col_A, col_B, col_C = st.columns(3)
                        
                        print("DEBUG: INITIAL - Displaying Chart A")
                        col_A.image(buf_A, width='stretch')
                        buf_A.close()
                        
                        print("DEBUG: INITIAL - Displaying Chart B")
                        col_B.image(buf_B, width='stretch')
                        buf_B.close()
                        
                        print("DEBUG: INITIAL - Displaying Chart C")
                        col_C.image(buf_C, width='stretch')
                        buf_C.close()
                        
                        print("DEBUG: INITIAL - All charts displayed successfully")

                            

                        # Série A — Demanda vs Atendimento vs Gap
                        df_demand_served = build_timeseries_demand_served(
                            need=need,
                            tasks_schedule=tasks_schedule,
                            slot_minutes=15
                        )

                        # Série B — Capacidade vs Utilização
                        df_capacity_util = build_timeseries_capacity_utilization(
                            workers_schedule=workers_schedule,
                            tasks_schedule=tasks_schedule,
                            cap_tasks_per_driver_per_slot=cap_tasks_per_driver_per_slot,
                            slot_minutes=15
                        )

                        # Série C — Déficit isolado
                        df_gap = build_timeseries_gap(
                            need=need,
                            tasks_schedule=tasks_schedule,
                            slot_minutes=15
                        )


                        import pandas as pd

                        with pd.ExcelWriter("timeseries_results.xlsx") as writer:
                            df_demand_served.to_excel(writer, sheet_name="Demand_vs_Served", index=False)
                            df_gap.to_excel(writer, sheet_name="Gap", index=False)
                            df_capacity_util.to_excel(writer, sheet_name="Capacity_Utilization", index=False)
                    except Exception as e:
                        st.error(f"Error rendering Initial charts: {e}")

                print("DEBUG: POST-INITIAL - About to render Results expander")
                #-----------------------------------
                # Tabela de Resultados
                # ----------------------------------            
                with st.expander("Results", expanded=True):
                    print("DEBUG: RESULTS - Inside expander")
                    col_A, col_B  = st.columns([5,2])
                    print("DEBUG: RESULTS - Columns created")
                    with col_A:
                        # Processar statisticsResult para separar descrições e valores
                        results = {
                            "Description": [],
                            "Value": []
                        }
                        
                        # Preencher o dicionário com os resultados
                        if statistics_result is None:
                            statistics_result = []
                        for stat in statistics_result:
                            # Separar a descrição e o valor usando ':'
                            if ':' in stat:
                                descricao, valor = stat.split(':', 1)  # Divide apenas na primeira ocorrência
                                results["Description"].append(descricao.strip())  # Adiciona a descrição sem espaços em branco
                                results["Value"].append(valor.strip())  # Adiciona o valor sem espaços em branco
                            else:
                                # Caso não haja ':' no stat, adicionar como descrição e valor em branco
                                results["Description"].append(stat)
                                results["Value"].append("")
                                
                        # Criar um DataFrame a partir do dicionário
                        results_df = pd.DataFrame(results)
                        
                        # Definir função para estilizar células
                        def highlight_cell(x):
                            
                            """Estiliza células específicas."""
                            df = pd.DataFrame('', index=x.index, columns=x.columns)  # Cria DataFrame vazio para estilos
                            
                            # Exemplo: pinta a célula onde Descrição é "Erro Grave" e Valor é -1
                            for i, row in x.iterrows():
                                if row["Description"] == "Model State" and row["Value"] == "OPTIMAL":
                                    df.loc[i, "Value"] = "background-color: green; color: white;"
                                elif row["Description"] == "Model State" and row["Value"] == "FEASIBLE":    
                                    df.loc[i, "Value"] = "background-color: orange; color: white;"
                                elif row["Description"] == "Model State" and (row["Value"] == "INFEASIBLE" 
                                                                            or row["Value"] == "UNBOUNDED"
                                                                            or row["Value"] == "ABNORMAL"
                                                                            or row["Value"] == "MODEL_INVALID"
                                                                            or row["Value"] == "Density not acceptable"):    
                                    df.loc[i, "Value"] = "background-color: red; color: white;"
                                elif row["Description"] == "Model State" and row["Value"] == "NOT_SOLVED":
                                    df.loc[i, "Value"] = "background-color: blue; color: white;"
                                elif row["Description"] == "Model Type":
                                    df.loc[i, "Value"] = "background-color: silver; color: white;"
                                    
                            return df

                        # Aplicar estilização
                        styled_df = results_df.style.apply(highlight_cell, axis=None)

                        # Exibir o DataFrame como tabela
                        st.table(styled_df)                                
                    
                    with col_B:
                        # ---------------------------------------------------------
                        # Radar chart
                        # ---------------------------------------------------------
                        kpis, coverage, safety_margin = compute_kpis(
                            demanda,
                            workers_schedule=workers_schedule,
                        )
                        
                        if optimization_mode in ["Exact", "Heuristic", "LNS"]:
                            radar = plot_kpi_radar({
                                "GlobalCoverage": kpis.get("global_coverage", 0.0),
                                "Efficiency": kpis.get("worker_efficiency", 0.0),
                                "Stability": kpis.get("temporal_stability", 0.0),
                                "Risk": 1 - kpis.get("operational_risk", 0.0),
                            })
                            safe_render_figure(radar)

                # ---------------------------------------------------------
                # Interpreter Panel
                # ---------------------------------------------------------
                with st.expander("Solver Intelligence & Interpretation", expanded=True):
                    try:
                        interp = interpret_solver_result(statistics_result)
                        
                        st.markdown(f"### 🧠 {t('interp_title')}")
                        st.success(interp.get("model_state", t("eng_unknown")))

                        col_met, col_str = st.columns(2)
                        with col_met:
                            st.markdown(f"##### 📊 {t('interp_op_eff')}")
                            st.json(interp.get("operational_metrics", {}))
                        
                        with col_str:
                            st.markdown(f"##### ⚙️ {t('interp_model_struct')}")
                            for line in interp.get("model_structure", []):
                                st.info(line)

                        st.markdown(f"### 🧮 {t('interp_service_quality')}")
                        for q in interp.get("solution_quality", []):
                            st.write("•", q)

                        st.markdown(f"### ⚙️ {t('interp_solver_behavior')}")
                        for line in interp.get("solver_behavior", []):
                            st.warning(line)

                    except Exception as e:
                        st.error(f"Interpreter Error: {e}")

                # ---------------------------------------------------------
                # Operational Effort Panel
                # ---------------------------------------------------------
                with st.expander("Driver Effort & Equity", expanded=True):
                    try:
                        col_A, col_B, col_C = st.columns([1.1, 1, 1])
                        
                        with col_A:
                            # 1) Painel de Equidade (Lorenz, Gini, ECDF)
                            if matrix_allocation.sum() > 0:
                                active_drivers_idx = [
                                    driver_idx for driver_idx in range(matrix_allocation.shape[1])
                                    if matrix_allocation[:, driver_idx].sum() > 0
                                ]
                                
                                # Carga de trabalho agregada em horas
                                slot_hours = 0.25 # TODO: use slot_minutes
                                horas_ativos = np.array([
                                    matrix_allocation[:, d].sum() * slot_hours 
                                    for d in active_drivers_idx
                                ])
                                
                                fig_equidade, resumo = plot_painel_esforco_operacional(
                                    horas_por_motorista=horas_ativos,
                                    limite_diario_horas=9.0
                                )
                                safe_render_figure(fig_equidade)
                            else:
                                st.info(t("interp_no_effort"))
                                resumo = {'motoristas_ativos': 0, 'media_horas': 0, 'mediana_horas': 0, 'percentil_90': 0, 'percentil_95': 0, 'indice_gini': 0}

                            st.caption(
                                t("interp_effort_caption").format(
                                    resumo['motoristas_ativos'],
                                    resumo['media_horas'],
                                    resumo['mediana_horas'],
                                    resumo['percentil_90'],
                                    resumo['percentil_95'],
                                    resumo['indice_gini']
                                )
                            )

                        with col_B:
                            if matrix_allocation.sum() > 0:
                                fig_eff, res_eff = plot_operational_effort_per_driver_active(
                                    matrix_allocation=matrix_allocation
                                )
                                safe_render_figure(fig_eff)
                            else:
                                st.info("No active drivers data.")

                        with col_C:
                            # Turno vs Condução
                            SHIFT_MAX_HOURS = 13
                            active_drivers_idx = [
                                driver_idx for driver_idx in range(matrix_allocation.shape[1])
                                if matrix_allocation[:, driver_idx].sum() > 0
                            ]

                            if driving_hours_per_driver is None:
                                driving_hours_per_driver = {}
                            elif isinstance(driving_hours_per_driver, list):
                                driving_hours_per_driver = {i: float(v) for i, v in enumerate(driving_hours_per_driver)}

                            driving_hours_active = [
                                driving_hours_per_driver.get(driver_idx, 0.0)
                                for driver_idx in active_drivers_idx
                            ]

                            slot_hours = 0.25
                            shift_hours_active = [
                                matrix_allocation[:, driver_idx].sum() * slot_hours
                                for driver_idx in active_drivers_idx
                            ]
                                                        
                            safe_render_figure(plot_panel_a_driving(driving_hours_active))
                            st.divider()
                            safe_render_figure(plot_panel_b_shift(shift_hours_active, SHIFT_MAX=SHIFT_MAX_HOURS))

                    except Exception as e:
                        st.error(f"Error rendering Effort panel: {e}")

                # ---------------------------------------------------------
                # Demand & KPI Analysis
                # ---------------------------------------------------------
                with st.expander("Deep Analysis & Metrics", expanded=True):
                    try:
                        col_A, col_B = st.columns(2)
                        
                        # Data Prep
                        slots = list(range(1, len(demanda) + 1))
                        df_analysis = pd.DataFrame({
                            "Slot": slots,
                            "Demanda": demanda,
                            "Motoristas": workers_schedule
                        })
                        df_analysis["Gap"] = df_analysis["Motoristas"] - df_analysis["Demanda"]
                        df_analysis["Violation"] = df_analysis["Gap"] < 0
                        
                        with col_A:
                            safe_render_figure(plot_demand_vs_capacity_2(df_analysis))
                        
                        with col_B:
                            safe_render_figure(plot_capacity_gap(df_analysis))

                        st.divider()
                        
                        col_heat, col_kpi = st.columns([1.5, 1])
                        
                        with col_heat:
                            if matrix_allocation is not None:
                                mat = np.asarray(matrix_allocation, dtype=int)
                                from matplotlib.colors import ListedColormap
                                from matplotlib.figure import Figure
                                import seaborn as sns

                                cmap = ListedColormap(["white", "#1f78b4"])
                                fig_heat = Figure(figsize=(14, FIG_HEIGHT))
                                ax_heat = fig_heat.subplots()

                                sns.heatmap(mat.T, ax=ax_heat, cmap=cmap, cbar=True)
                                ax_heat.set_xlabel("Slots")
                                ax_heat.set_ylabel("Drivers")
                                ax_heat.set_title("Allocation Map")
                                safe_render_figure(fig_heat)
                            else:
                                st.info("No allocation map available.")

                        with col_kpi:
                            kpis, coverage, safety_margin = compute_kpis(
                                demanda,
                                workers_schedule=workers_schedule,
                                matrix_allocation=matrix_allocation,
                            )
                            # Display top metrics
                            def safe_fmt(val):
                                return f"{val:.2%}" if val is not None else "0.00%"
                                
                            st.metric("Global Coverage", safe_fmt(kpis.get('global_coverage')))
                            st.metric("Efficiency", safe_fmt(kpis.get('worker_efficiency')))
                            st.metric("Operational Risk", safe_fmt(kpis.get('operational_risk')))
                            
                    except Exception as e:
                        st.error(f"Error rendering Analysis panel: {e}")
                  
                  
                with st.expander("LNS", expanded=True): 
                    # col_A__ = st.columns(1)
                    # with col_A__:
                    col_A, col_B, col_C = st.columns(3)
                    with col_A:
                        # Converter dados para DataFrame
                        if iterations_data_result != []:
                            # st.subheader("Convergence Progress")
                            df_iterationsResult = pd.DataFrame(iterations_data_result)
                            
                            # 🔒 Garantir colunas mínimas esperadas
                            if "objective_value" not in df_iterationsResult.columns:
                                df_iterationsResult["objective_value"] = np.nan

                            if "iteration" not in df_iterationsResult.columns:
                                df_iterationsResult["iteration"] = range(len(df_iterationsResult))
                            
                            if "relaxation_level" not in df_iterationsResult.columns:
                                df_iterationsResult["relaxation_level"] = range(len(df_iterationsResult))

                            # # Gráfico de convergência do objetivo
                            from matplotlib.figure import Figure
                            fig = Figure(figsize=(14, FIG_HEIGHT))
                            ax = fig.subplots()

                            if not df_iterationsResult.empty and df_iterationsResult["objective_value"].notna().any():
                                df_iterationsResult.plot(
                                    x="iteration",
                                    y="objective_value",
                                    ax=ax,
                                    marker="o",
                                    label="Goal Value"
                                )
                            else:
                                ax.text(
                                    0.5, 0.5,
                                    "No objective values recorded",
                                    ha="center", va="center",
                                    transform=ax.transAxes
                                )

                            ax.set_title("Convergence of Result")
                with st.expander("LNS", expanded=True): 
                    if iterations_data_result != []:
                        df_iterationsResult = pd.DataFrame(iterations_data_result)
                        
                        # 🔒 Standardize columns
                        for col in ["objective_value", "iteration", "relaxation_level"]:
                            if col not in df_iterationsResult.columns:
                                df_iterationsResult[col] = np.nan
                        if "iteration" not in df_iterationsResult.columns:
                            df_iterationsResult["iteration"] = range(len(df_iterationsResult))

                        col_A, col_B, col_C = st.columns(3)
                        
                        with col_A:
                            # 1) Convergence chart
                            from matplotlib.figure import Figure
                            fig_conv = Figure(figsize=(14, FIG_HEIGHT))
                            ax_conv = fig_conv.subplots()

                            if not df_iterationsResult.empty and df_iterationsResult["objective_value"].notna().any():
                                df_iterationsResult.plot(
                                    x="iteration",
                                    y="objective_value",
                                    ax=ax_conv,
                                    marker="o",
                                    label="Goal Value"
                                )
                            else:
                                ax_conv.text(0.5, 0.5, "No objective values", ha="center", va="center", transform=ax_conv.transAxes)

                            ax_conv.set_title("Convergence of Result")
                            ax_conv.set_xlabel("Iteration")
                            ax_conv.set_ylabel("Goal Value")
                            ax_conv.grid(True)
                            ax_conv.legend()
                            safe_render_figure(fig_conv)

                        with col_B:
                            # 2) Improvement chart
                            if "improved" in df_iterationsResult.columns:
                                fig_imp = Figure(figsize=(14, FIG_HEIGHT))
                                ax_imp = fig_imp.subplots()

                                imp_int = df_iterationsResult["improved"].astype(int)
                                ax_imp.bar(
                                    df_iterationsResult["iteration"],
                                    imp_int,
                                    color=["green" if x == 1 else "lightgray" for x in imp_int]
                                )
                                ax_imp.set_title("LNS – Improvement per Iteration")
                                ax_imp.set_xlabel("Iteration")
                                ax_imp.set_ylabel("Improved (1 = Yes)")
                                ax_imp.set_yticks([0, 1])
                                ax_imp.grid(True)
                                safe_render_figure(fig_imp)

                        with col_C:
                            # 3) Workers Evolution
                            if "total_workers" in df_iterationsResult.columns:
                                fig_wrk = Figure(figsize=(14, FIG_HEIGHT))
                                ax_wrk = fig_wrk.subplots()

                                df_iterationsResult.plot(
                                    x="iteration",
                                    y="total_workers",
                                    ax=ax_wrk,
                                    marker="s",
                                    color="purple",
                                    label="Total Workers"
                                )
                                ax_wrk.set_title("LNS – Total Workers Evolution")
                                ax_wrk.set_xlabel("Iteration")
                                ax_wrk.set_ylabel("Total Workers")
                                ax_wrk.grid(True)
                                ax_wrk.legend()
                                safe_render_figure(fig_wrk)

                            # 4) Relaxation Level
                            if not df_iterationsResult["relaxation_level"].isna().all():
                                fig_relax = Figure(figsize=(14, FIG_HEIGHT))
                                ax_relax = fig_relax.subplots()

                                df_iterationsResult.plot(
                                    x="iteration",
                                    y="relaxation_level",
                                    ax=ax_relax,
                                    marker="x",
                                    color="red",
                                    label="Relaxation Level"
                                )
                                ax_relax.set_title("Relaxation Progress")
                                ax_relax.set_xlabel("Iteration")
                                ax_relax.set_ylabel("Relaxation Level")
                                ax_relax.grid(True)
                                ax_relax.legend()
                                safe_render_figure(fig_relax)
                    
                    # with col_B__:
                    if iterations_data_result != []:
                        # Fix ArrowTypeError: convert columns with mixed types to string
                        if "status" in df_iterationsResult.columns:
                            df_iterationsResult["status"] = df_iterationsResult["status"].astype(str)
                        st.table(df_iterationsResult)  
                        # if iterations_data_result != []:
                        #                 st.subheader("Relaxation Progress")
                        #                 fig_relax, ax_relax = plt.subplots(figsize=(14, FIG_HEIGHT))
                        #                 df_iterationsResult.plot(x="iteration", y="relaxation_level", ax=ax_relax, marker="x", color="red", label="Relaxation Level")
                        #                 ax_relax.set_title("Relaxation Progress")
                        #                 ax_relax.set_xlabel("Iteration")
                        #                 ax_relax.set_ylabel("Relaxation Level")
                        #                 ax_relax.grid(True)
                        #                 ax_relax.legend()

                        #                 # Exibir o gráfico no Streamlit
                        #                 st.pyplot(fig_relax)                    
                    
                                          
                # ---------------------------------------------------------
                # Matrix
                # ---------------------------------------------------------
                with st.expander("Matrix", expanded=True): 
                    col_resultsIniI, col_resultsIniII = st.columns(2)
                    with col_resultsIniI:
                        if msg is not None:    
                            if initial_density_matrix is not None:
                                # Exibir a densidade
                                st.write(f"Density: {initial_density_matrix:.4f}")
                                fig, ax = plt.subplots(figsize=(14, FIG_HEIGHT))
                                sns.heatmap(initial_constraints_coefficients, cmap="Blues", cbar=False, annot=False, fmt="d", annot_kws={"size": 7})
                                plt.title('Constraint Matrix')
                                plt.xlabel('X')
                                plt.ylabel('Period')
                                safe_render_figure(fig)
                            
                        #     # Converter dados para DataFrame
                        # if iterations_data_result != []:
                        #     st.subheader("Convergence Progress")
                        #     df_iterationsResult = pd.DataFrame(iterations_data_result)
                            
                        #     # 🔒 Garantir colunas mínimas esperadas
                        #     if "objective_value" not in df_iterationsResult.columns:
                        #         df_iterationsResult["objective_value"] = np.nan

                        #     if "iteration" not in df_iterationsResult.columns:
                        #         df_iterationsResult["iteration"] = range(len(df_iterationsResult))
                            
                        #     if "relaxation_level" not in df_iterationsResult.columns:
                        #         df_iterationsResult["relaxation_level"] = range(len(df_iterationsResult))

                        #     # # Gráfico de convergência do objetivo
                        #     fig, ax = plt.subplots(figsize=(14, FIG_HEIGHT))
                        #     if not df_iterationsResult.empty and df_iterationsResult["objective_value"].notna().any():
                        #         df_iterationsResult.plot(
                        #             x="iteration",
                        #             y="objective_value",
                        #             ax=ax,
                        #             marker="o",
                        #             label="Goal Value"
                        #         )
                        #     else:
                        #         ax.text(
                        #             0.5, 0.5,
                        #             "No objective values recorded",
                        #             ha="center", va="center",
                        #             transform=ax.transAxes
                        #         )

                        #     ax.set_title("Convergence of Result")
                        #     ax.set_xlabel("Iteration")
                        #     ax.set_ylabel("Goal Value")
                        #     ax.grid(True)
                        #     ax.legend()
                            
                        #     # Exibir o gráfico no Streamlit
                        #     st.pyplot(fig)    

                            # if "improved" in df_iterationsResult.columns:
                            #     fig, ax = plt.subplots(figsize=(14, FIG_HEIGHT))

                            #     df_iterationsResult["improved_int"] = df_iterationsResult["improved"].astype(int)

                            #     ax.bar(
                            #         df_iterationsResult["iteration"],
                            #         df_iterationsResult["improved_int"],
                            #         color=["green" if x == 1 else "lightgray" for x in df_iterationsResult["improved_int"]]
                            #     )
                            #     st.subheader("LNS – Improvement")
                            #     ax.set_title("LNS – Improvement per Iteration")
                            #     ax.set_xlabel("Iteration")
                            #     ax.set_ylabel("Improved (1 = Yes)")
                            #     ax.set_yticks([0, 1])
                            #     ax.grid(True)

                            #     st.pyplot(fig)
                                
                            # if "total_workers" in df_iterationsResult.columns:
                            #     fig, ax = plt.subplots(figsize=(14, FIG_HEIGHT))

                            #     df_iterationsResult.plot(
                            #         x="iteration",
                            #         y="total_workers",
                            #         ax=ax,
                            #         marker="s",
                            #         color="purple",
                            #         label="Total Workers"
                            #     )

                            #     st.subheader("LNS – Workers Evolution")
                            #     ax.set_title("LNS – Total Workers Evolution")
                            #     ax.set_xlabel("Iteration")
                            #     ax.set_ylabel("Total Workers")
                            #     ax.grid(True)
                            #     ax.legend()

                            #     st.pyplot(fig)

                    with col_resultsIniII:
                        if msg is not None:
                            if final_density is not None:
                                st.write(f"Final Density Matrix Constraints: {final_density:.4f}")
                            else:
                                st.write("Final Density Matrix Constraints: not computed for this mode.")
                            fig, ax = plt.subplots(figsize=(14, FIG_HEIGHT))
                            constraints_coefficients = load_data('constraints_coefficients.json')
                            sns.heatmap(constraints_coefficients, cmap="Oranges", cbar=False, annot=False, fmt="d", annot_kws={"size": 6})
                            plt.title('Constraints Matrix')
                            plt.xlabel('X')
                            plt.ylabel('Period')
                            safe_render_figure(fig)
                                
                        # if iterations_data_result != []:
                        #                 st.subheader("Relaxation Progress")
                        #                 fig_relax, ax_relax = plt.subplots(figsize=(14, FIG_HEIGHT))
                        #                 df_iterationsResult.plot(x="iteration", y="relaxation_level", ax=ax_relax, marker="x", color="red", label="Relaxation Level")
                        #                 ax_relax.set_title("Relaxation Progress")
                        #                 ax_relax.set_xlabel("Iteration")
                        #                 ax_relax.set_ylabel("Relaxation Level")
                        #                 ax_relax.grid(True)
                        #                 ax_relax.legend()

                        #                 # Exibir o gráfico no Streamlit
                        #                 st.pyplot(fig_relax)                    
                    
                    # if iterations_data_result != []:            
                    #     st.table(df_iterationsResult)
                    if msg is not None:
                        # Criando o DataFrame a partir da lista msgResult
                        results_msg_result = {
                            "Description": [],
                            "Value": []
                        }
                        # Preencher o dicionário com os resultados
                        for stat in msg:
                            # Separar a descrição e o valor usando ':'
                            if ':' in stat:
                                descricao, valor = stat.split(':', 1)  # Divide apenas na primeira ocorrência
                                results_msg_result["Description"].append(descricao.strip())  # Adiciona a descrição sem espaços em branco
                                results_msg_result["Value"].append(valor.strip())  # Adiciona o valor sem espaços em branco
                            else:
                                # Caso não haja ':' no stat, adicionar como descrição e valor em branco
                                results_msg_result["Description"].append(stat)
                                results_msg_result["Value"].append("")
                        # Criar um DataFrame a partir do dicionário
                        dfmsgResult = pd.DataFrame(results_msg_result)
                        # Exibir o DataFrame como tabela
                        st.table(dfmsgResult)
                
                # ---------------------------------------------------------
                # Heatmaps analíticos
                # ---------------------------------------------------------
                with st.expander("Analytical Heatmaps", expanded=True):
                # KPI executive summary
                    colA1, colB1 = st.columns(2)
                    with colA1:
                        coverage_vec = kpis.get("coverage")

                        if (
                            isinstance(coverage_vec, np.ndarray)
                            and coverage_vec.ndim == 1
                            and coverage_vec.size > 0
                        ):
                            heat1 = plot_heatmap_coverage(coverage_vec)
                            safe_render_figure(heat1)
                        else:
                            st.info("Coverage heatmap indisponível para esta solução.")            
                    with colB1:
                        
                        safety_vec = kpis.get("safety_margin")

                        if (
                            isinstance(safety_vec, np.ndarray)
                            and safety_vec.ndim == 1
                            and safety_vec.size > 0
                        ):
                            heat2 = plot_heatmap_safety(safety_vec)
                            safe_render_figure(heat2)
                        else:
                            st.info("Safety margin heatmap indisponível para esta solução.")            
                    
                with st.expander("Solver Logs", expanded=False):
                    st.subheader("Standard Output")

                    try:
                        stdout_text = solver_logs.get("stdout", "") if isinstance(solver_logs, dict) else str(solver_logs)
                        stderr_text = solver_logs.get("stderr", "") if isinstance(solver_logs, dict) else ""
                    except Exception:
                        stdout_text = str(solver_logs)
                        stderr_text = ""

                    # Escapar caracteres problemáticos para evitar erros de parser
                    stdout_text = stdout_text.replace("<", "&lt;").replace(">", "&gt;")
                    stderr_text = stderr_text.replace("<", "&lt;").replace(">", "&gt;")

                    st.text_area("stdout", stdout_text, height=250)

                    st.subheader("Error Output")
                    st.text_area("stderr", stderr_text, height=250)

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            st.code(traceback.format_exc())
    
