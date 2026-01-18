# Caminho absoluto para a pasta "simulator"

import os
import sys
from typing import Any, List, Optional, Tuple

BASE_SIMULATOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Garante que o Python reconhece "ml/" como pacote
if BASE_SIMULATOR_DIR not in sys.path:
    sys.path.append(BASE_SIMULATOR_DIR)

import html
import math
import traceback
import streamlit as st
from ortools.linear_solver import pywraplp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from math import ceil
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import io, contextlib

from heuristic import greedy_initial_allocation
from lns import run_lns
from ml.ml_guidance import assignment_scorer, neighborhood_scorer
from dataset_builder import build_f1_rows, build_f2_rows, save_datasets
from ml.trainer import train_all_models
import traceback
import time
from kpis import compute_kpis
from charts import (
    plot_heatmap_safety,
    plot_heatmap_coverage,
    plot_kpi_radar,
)


import html
# Habilitar a largura total da página
st.set_page_config(layout="wide")


#---------------------------------
# Graficos
#--------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_panel_a_driving(driving_hours_active):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.sort(np.asarray(driving_hours_active))
    y = np.arange(1, len(x)+1) / len(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ECDF
    axes[0].plot(x, y, linewidth=2)
    axes[0].axvline(9.0, linestyle="--")
    axes[0].set_title("ECDF — Condução (UE)")
    axes[0].set_xlabel("Horas de Condução")
    axes[0].set_ylabel("Proporção acumulada")
    axes[0].grid(alpha=0.3)

    # Boxplot
    axes[1].boxplot(x, vert=True)
    axes[1].axhline(9.0, linestyle="--")
    axes[1].set_title("Boxplot — Condução (UE)")
    axes[1].set_ylabel("Horas")
    axes[1].grid(alpha=0.3)

    return fig

def plot_panel_b_shift(shift_hours_active, SHIFT_MAX=13):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.sort(np.asarray(shift_hours_active))
    y = np.arange(1, len(x)+1) / len(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ECDF
    axes[0].plot(x, y, linewidth=2)
    axes[0].axvline(SHIFT_MAX, linestyle="--")
    axes[0].set_title("ECDF — Turno (Presença)")
    axes[0].set_xlabel("Horas de Turno")
    axes[0].set_ylabel("Proporção acumulada")
    axes[0].grid(alpha=0.3)

    # Boxplot
    axes[1].boxplot(x, vert=True)
    axes[1].axhline(SHIFT_MAX, linestyle="--")
    axes[1].set_title("Boxplot — Turno (Presença)")
    axes[1].set_ylabel("Horas")
    axes[1].grid(alpha=0.3)

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

    df = pd.DataFrame({
        "Slot": np.arange(1, len(demanda) + 1),
        "Demanda": demanda,
        "Motoristas": workers_schedule
    })

    df["Gap"] = df["Motoristas"] - df["Demanda"]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.axhline(0, color="black", linewidth=1)
    ax.bar(
        df["Slot"],
        df["Gap"],
        color=df["Gap"].apply(lambda x: "green" if x >= 0 else "red"),
        alpha=0.8
    )

    ax.set_title("Gap de Cobertura por Período (Motoristas − Demanda)")
    ax.set_xlabel("Período (slot)")
    ax.set_ylabel("Gap de Cobertura")

    # Métricas auxiliares (úteis para o texto)
    violations = (df["Gap"] < 0).sum()
    min_gap = df["Gap"].min()

    ax.text(
        0.01, 0.95,
        f"Slots com violação: {violations}\nPior gap: {min_gap}",
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
    import numpy as np
    import matplotlib.pyplot as plt

    demanda = np.asarray(demanda, dtype=float)
    workers_schedule = np.asarray(workers_schedule, dtype=float)
    tasks_schedule = np.asarray(tasks_schedule, dtype=float)

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

    fig, ax1 = plt.subplots(figsize=(14, 5))

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

    ax1.set_xlabel("Período (slot)")
    ax1.set_ylabel("Tarefas")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        slots,
        utilizacao,
        label="Utilização da Capacidade",
        linestyle=":",
        linewidth=2
    )
    ax2.set_ylabel("Utilização (0–1)")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, linestyle="--", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Demanda vs Capacidade vs Atendimento Real")

    return fig


# def plot_demand_capacity_utilization(
#     demanda,
#     drivers_present_per_slot,     # ← derivado de Y
#     tasks_served_per_slot,        # ← derivado de X
#     cap_tasks_per_driver_per_slot
# ):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # --- Sanitização ---
#     demanda = np.asarray(demanda, dtype=float)
#     drivers_present_per_slot = np.asarray(drivers_present_per_slot, dtype=float)
#     tasks_served_per_slot = np.asarray(tasks_served_per_slot, dtype=float)

#     slots = np.arange(len(demanda))

#     # --- Capacidade real ---
#     capacidade_por_slot = drivers_present_per_slot * cap_tasks_per_driver_per_slot

#     # --- Utilização real ---
#     utilizacao = np.divide(
#         tasks_served_per_slot,
#         capacidade_por_slot,
#         out=np.zeros_like(tasks_served_per_slot, dtype=float),
#         where=capacidade_por_slot > 0
#     )

#     # --- Plot ---
#     fig, ax1 = plt.subplots(figsize=(14, 5))

#     ax1.plot(slots, demanda, label="Demanda (tarefas)", linewidth=2)
#     ax1.plot(
#         slots,
#         capacidade_por_slot,
#         label="Capacidade Alocada (tarefas)",
#         linewidth=2,
#         alpha=0.9
#     )

#     ax1.set_xlabel("Período (slot)")
#     ax1.set_ylabel("Tarefas")
#     ax1.grid(alpha=0.3)

#     # --- Segundo eixo: utilização ---
#     ax2 = ax1.twinx()
#     ax2.plot(
#         slots,
#         utilizacao,
#         label="Utilização da Capacidade",
#         linestyle="--",
#         linewidth=2
#     )
#     ax2.set_ylabel("Utilização (0–1)")
#     ax2.set_ylim(0, 1.05)
#     ax2.axhline(1.0, linestyle=":", alpha=0.6)

#     # --- Legenda combinada ---
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#     ax1.set_title("Demanda vs Capacidade Alocada e Utilização Real")

#     return fig

# def plot_demand_capacity_utilization(
#     demanda,
#     matrix_allocation,
#     cap_tasks_per_driver_per_slot
# ):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # --- Sanitização ---
#     demanda = np.asarray(demanda, dtype=float)
#     matrix_allocation = np.asarray(matrix_allocation, dtype=int)

#     slots = np.arange(len(demanda))

#     # --- Capacidade ---
#     motoristas_por_slot = matrix_allocation.sum(axis=1)
#     capacidade_por_slot = motoristas_por_slot * cap_tasks_per_driver_per_slot

#     # --- Utilização real da capacidade ---
#     demanda_atendida = np.minimum(demanda, capacidade_por_slot)

#     utilizacao = np.divide(
#         demanda_atendida,
#         capacidade_por_slot,
#         out=np.zeros_like(demanda_atendida, dtype=float),
#         where=capacidade_por_slot > 0
#     )

#     # --- Plot ---
#     fig, ax1 = plt.subplots(figsize=(14, 5))

#     # Eixo Y principal — quantidades absolutas
#     ax1.plot(slots, demanda, label="Demanda (tarefas)", linewidth=2, color="tab:blue")
#     ax1.plot(
#         slots,
#         capacidade_por_slot,
#         label="Capacidade Alocada (tarefas)",
#         linewidth=2,
#         color="tab:orange",
#         alpha=0.9
#     )

#     ax1.set_xlabel("Período (slot)")
#     ax1.set_ylabel("Tarefas")
#     ax1.grid(alpha=0.3)

#     # --- Segundo eixo — utilização ---
#     ax2 = ax1.twinx()
#     ax2.plot(
#         slots,
#         utilizacao,
#         label="Utilização da Capacidade",
#         linestyle="--",
#         linewidth=2,
#         color="tab:green"
#     )
#     ax2.set_ylabel("Utilização (0–1)")
#     ax2.set_ylim(0, 1.05)

#     # Linha de referência (100%)
#     ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.6)

#     # --- Legenda combinada ---
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(
#         lines1 + lines2,
#         labels1 + labels2,
#         loc="upper left"
#     )

#     ax1.set_title("Demanda vs Capacidade Alocada e Utilização da Capacidade")

#     return fig

# def plot_demand_vs_capacity(
#     demanda,
#     drivers_present_per_slot,
#     tasks_served_per_slot,
#     cap_tasks_per_driver_per_slot
# ):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     demanda = np.asarray(demanda, dtype=float)
#     drivers_present_per_slot = np.asarray(drivers_present_per_slot, dtype=float)
#     tasks_served_per_slot = np.asarray(tasks_served_per_slot, dtype=float)

#     capacidade = drivers_present_per_slot * cap_tasks_per_driver_per_slot
#     slots = np.arange(len(demanda))

#     fig, ax = plt.subplots(figsize=(14, 4))

#     ax.plot(slots, demanda, label="Demanda", linewidth=2)
#     ax.plot(slots, capacidade, label="Capacidade Alocada", linewidth=2)
#     ax.plot(
#         slots,
#         tasks_served_per_slot,
#         label="Atendimento Real",
#         linewidth=2,
#         linestyle="--"
#     )

#     ax.set_xlabel("Período (slot)")
#     ax.set_ylabel("Tarefas")
#     ax.set_title("Demanda vs Capacidade vs Atendimento Real")
#     ax.legend()
#     ax.grid(alpha=0.3)

#     return fig

def plot_demand_vs_capacity(
    demanda,
    workers_schedule,                 # presença (Y agregada)
    tasks_schedule,                   # atendimento real (X agregado)
    cap_tasks_per_driver_per_slot
):
    import numpy as np
    import matplotlib.pyplot as plt

    demanda = np.asarray(demanda, dtype=float)
    workers_schedule = np.asarray(workers_schedule, dtype=float)
    tasks_schedule = np.asarray(tasks_schedule, dtype=float)

    capacidade = workers_schedule * cap_tasks_per_driver_per_slot
    slots = np.arange(len(demanda))

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(slots, demanda, label="Demanda", linewidth=2)
    ax.plot(slots, capacidade, label="Capacidade Alocada", linewidth=2)
    ax.plot(slots, tasks_schedule, label="Atendimento Real", linewidth=2, linestyle="--")

    ax.set_xlabel("Período (slot)")
    ax.set_ylabel("Tarefas")
    ax.set_title("Demanda vs Capacidade vs Atendimento Real")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


# def plot_demand_vs_capacity(demanda, matrix_allocation, cap_tasks_per_driver_per_slot):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     demanda = np.asarray(demanda, dtype=float)
#     matrix_allocation = np.asarray(matrix_allocation, dtype=int)

#     motoristas_por_slot = matrix_allocation.sum(axis=1)
#     capacidade = motoristas_por_slot * cap_tasks_per_driver_per_slot
#     slots = np.arange(len(demanda))

#     fig, ax = plt.subplots(figsize=(14, 4))

#     ax.plot(slots, demanda, label="Demanda", linewidth=2)
#     ax.plot(slots, capacidade, label="Capacidade Alocada", linewidth=2)

#     ax.set_xlabel("Período (slot)")
#     ax.set_ylabel("Tarefas")
#     ax.set_title("Demanda vs Capacidade Alocada")
#     ax.legend()
#     ax.grid(alpha=0.3)

#     return fig

def plot_capacity_utilization(demanda, matrix_allocation, cap_tasks_per_driver_per_slot):
    import numpy as np
    import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(slots, utilizacao, linestyle="--", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Período (slot)")
    ax.set_ylabel("Utilização")
    ax.set_title("Utilização da Capacidade Alocada")
    ax.grid(alpha=0.3)

    return fig


# def plot_demand_capacity_utilization(
#     demanda,
#     matrix_allocation,
#     cap_tasks_per_driver_per_slot
# ):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     demanda = np.asarray(demanda, dtype=float)
#     matrix_allocation = np.asarray(matrix_allocation, dtype=int)

#     motoristas_por_slot = matrix_allocation.sum(axis=1)

#     capacidade_por_slot = motoristas_por_slot * cap_tasks_per_driver_per_slot

#     demanda_atendida = np.minimum(demanda, capacidade_por_slot)

#     utilizacao = np.divide(
#         demanda_atendida,
#         capacidade_por_slot,
#         out=np.zeros_like(demanda, dtype=float),
#         where=capacidade_por_slot > 0
#     )

#     slots = np.arange(len(demanda))

#     fig, ax = plt.subplots(figsize=(14, 5))

#     ax.plot(slots, demanda, label="Demanda (tarefas)", linewidth=2)
#     ax.plot(slots, capacidade_por_slot, label="Capacidade Alocada (tarefas)", linewidth=2)
#     ax.plot(slots, utilizacao, label="Utilização da Capacidade", linestyle="--")

#     ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6)

#     ax.set_title("Demanda vs Capacidade Alocada vs Utilização")
#     ax.set_xlabel("Período (slot)")
#     ax.set_ylabel("Tarefas / Utilização")
#     ax.legend()
#     ax.grid(alpha=0.3)

#     return fig

# def plot_demand_capacity_utilization(
#     demanda,
#     matrix_allocation,
#     cap_tasks_per_driver_per_slot
# ):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     num_periods = len(demanda)

#     motoristas_por_slot = matrix_allocation.sum(axis=1)
#     capacidade_por_slot = motoristas_por_slot * cap_tasks_per_driver_per_slot

#     utilizacao_media = np.divide(
#         capacidade_por_slot,
#         motoristas_por_slot,
#         out=np.zeros_like(capacidade_por_slot, dtype=float),
#         where=motoristas_por_slot > 0
#     )

#     slots = np.arange(num_periods)

#     fig, ax = plt.subplots(figsize=(14, 5))

#     ax.plot(slots, demanda, label="Demanda (tarefas)", linewidth=2)
#     ax.plot(slots, capacidade_por_slot, label="Capacidade Alocada", linewidth=2)
#     ax.plot(slots, utilizacao_media, label="Utilização média por motorista", linestyle="--")

#     ax.set_title("Demanda vs Capacidade vs Utilização Média")
#     ax.set_xlabel("Período (slot)")
#     ax.set_ylabel("Quantidade")
#     ax.legend()
#     ax.grid(alpha=0.3)

#     return fig


def plot_demand_vs_drivers_line(demanda, workers_schedule):
    df = pd.DataFrame({
        "Slot": np.arange(1, len(demanda) + 1),
        "Demanda": demanda,
        "Motoristas": workers_schedule
    })

    fig, ax = plt.subplots(figsize=(14, 5))

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
    title: str = "Distribuição do Esforço Operacional por Motorista",
):
    """
    Gráfico 1 — Esforço Operacional por Motorista

    Interpretação:
    - NÃO é objetivo do modelo
    - Representa esforço operacional total
    - Usado para análise de eficiência e coerência regulatória
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    if matrix_allocation is None:
        st.info("Gráfico indisponível: matrix_allocation é None.")
        return

    mat = np.asarray(matrix_allocation, dtype=int)

    if mat.ndim != 2 or mat.size == 0:
        st.info("Gráfico indisponível: matriz de alocação inválida.")
        return

    # -----------------------------
    # Cálculos
    # -----------------------------
    slot_hours = slot_minutes / 60.0

    slots_per_driver = mat.sum(axis=0)
    
    # 🔥 AJUSTE FUNDAMENTAL: filtrar apenas motoristas ativos
    active_mask = slots_per_driver > 0
    slots_active_drivers = slots_per_driver[active_mask]

    hours_per_driver = slots_active_drivers * slot_hours    
    
    # hours_per_driver = slots_per_driver * slot_hours

    total_assigned_slots = int(slots_per_driver.sum())
    total_hours = total_assigned_slots * slot_hours

    avg_hours = float(hours_per_driver.mean()) if len(hours_per_driver) else 0.0

    # -----------------------------
    # Gráfico
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        range(1, len(hours_per_driver) + 1),
        hours_per_driver,
        alpha=0.85,
        label="Horas trabalhadas"
    )

    ax.axhline(
        daily_limit_hours,
        color="red",        
        linestyle="--",
        linewidth=2,
        label=f"Limite diário legal ({daily_limit_hours}h)"
    )

    ax.axhline(
        avg_hours,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Média observada ({avg_hours:.2f}h)"
    )

    ax.set_xlabel("Motorista")
    ax.set_ylabel("Horas trabalhadas no horizonte")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    st.pyplot(fig)

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

    Parâmetros
    ----------
    matrix_allocation : np.ndarray
        Matriz binária [períodos x motoristas]
    slot_minutes : int
        Duração de cada slot (padrão = 15 minutos)
    limite_diario_horas : float
        Limite diário legal de condução (padrão = 9h – Reg. 561/2006)

    Retorno
    -------
    fig : matplotlib.figure.Figure
    resumo : dict
        Métricas consolidadas para análise e dissertação
    """

    if matrix_allocation is None or matrix_allocation.size == 0:
        raise ValueError("matrix_allocation inválida ou vazia.")

    # ------------------------------------------------
    # 1) Carga por motorista (em slots)
    # ------------------------------------------------
    slots_por_motorista = matrix_allocation.sum(axis=0)

    # 🔴 AJUSTE FUNDAMENTAL:
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
    fig, ax = plt.subplots(figsize=(10, 4))
    
    horas_por_motorista = horas_por_motorista #np.sort(horas_por_motorista)

    ax.bar(
        range(1, total_motoristas_ativos + 1),
        horas_por_motorista,
        color="steelblue",
        alpha=0.85,
        label="Carga atribuída (horas)"
    )

    # Linha de referência legal
    ax.axhline(
        limite_diario_horas,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Limite diário legal ({limite_diario_horas}h)"
    )


    ax.axhline(
        media_horas,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Média observada ({media_horas:.2f}h)"
    )
    
    
    # ax.set_xlabel("Motoristas ativos")
    # ax.set_xlabel("Motoristas ativos (ordem sem significado semântico)")
    ax.set_xlabel("Motoristas ativos (ordenados por carga)")
    ax.set_ylabel("Carga de trabalho (horas)")
    ax.set_title("Distribuição do Esforço Operacional — Motoristas Ativos")

    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # ------------------------------------------------
    # 4) Texto explicativo (essencial para a dissertação)
    # ------------------------------------------------
    texto = (
        f"Total de slots atribuídos: {total_slots_atribuidos}\n"
        f"Total de horas de trabalho: {total_horas:.2f} h\n"
        f"Motoristas ativos: {total_motoristas_ativos}\n"
        f"Esforço médio: {media_horas:.2f} h por motorista\n\n"
        "Nota: Este indicador representa apenas\n"
        "o esforço operacional da solução,\n"
        "não sendo o objetivo principal\n"
        "do modelo de otimização."
        
        

    )
    
    

    # ax.text(
    #     1.02,
    #     0.5,
    #     texto,
    #     transform=ax.transAxes,
    #     fontsize=9,
    #     va="center",
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.95)
    # )

    # plt.tight_layout()

    # ------------------------------------------------
    # 5) Resumo estruturado (para relatório/dissertação)
    # ------------------------------------------------
    # resumo = {
    #     "total_slots_atribuidos": total_slots_atribuidos,
    #     "total_horas_trabalho": total_horas,
    #     "total_motoristas_ativos": total_motoristas_ativos,
    #     "media_horas_por_motorista_ativo": media_horas,
    #     "duracao_slot_minutos": slot_minutes,
    #     "limite_diario_horas": limite_diario_horas,
    # }
    
    st.pyplot(fig)
    
    
    # -----------------------------
    # Texto interpretativo
    # -----------------------------
    st.caption(
        f"""
        Total de slots atribuídos: **{total_assigned_slots}**  
        Duração por slot: **{slot_minutes} minutos**  
        Total de horas trabalhadas: **{total_hours:.2f} h**

        Em média, cada motorista trabalhou aproximadamente **{media_horas:.2f} h**.
        considerando a flexibilidade residual decorrente da não ativação do
        repouso diário mínimo neste cenário.

        ⚠️ **Importante:** este indicador representa **esforço operacional**.
        Ele **não é o objetivo do modelo**, mas um resultado emergente utilizado
        para análise de eficiência e validação regulatória.
        """
        
    )
    
    # st.caption(resumo)

    # return fig, resumo

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


def interpret_solver_result(result):
    # aceita dict OU list[str]
    if isinstance(result, list):
        result = parse_statistics_list(result)
    if not isinstance(result, dict):
        raise TypeError("interpret_solver_result espera dict ou list[str].")

    interpretation = {}

    status = result.get("status")
    gap = result.get("gap")

    if status == pywraplp.Solver.OPTIMAL and (gap is not None and gap <= 0.01):
        state_msg = "Solução ótima comprovada (gap ≤ 1%)."
    elif status == pywraplp.Solver.OPTIMAL:
        state_msg = f"Solução com prova de otimalidade no critério do solver (gap ~ {gap*100:.2f}% se disponível)."
    elif status == pywraplp.Solver.FEASIBLE:
        state_msg = "Solução viável encontrada, sem prova de otimalidade."
    elif status == pywraplp.Solver.INFEASIBLE:
        state_msg = "Modelo inviável com as restrições atuais."
    else:
        state_msg = "Estado do solver não identificado no resumo."

    interpretation["model_state"] = state_msg

    # Métricas operacionais
    assigned = result.get("total_assigned_slots") or 0
    drivers = result.get("total_active_drivers") or 0
    total_hours = assigned * 0.25
    avg_hours = total_hours / drivers if drivers > 0 else 0.0

    interpretation["operational_metrics"] = {
        "drivers_used": drivers,
        "assigned_slots": assigned,
        "total_hours": round(total_hours, 2),
        "avg_hours_per_driver": round(avg_hours, 2),
    }

    # Densidade
    density = result.get("density")
    thr = result.get("density_threshold")
    if density is not None and thr is not None:
        interpretation["density_note"] = f"Densidade final={density:.4f} vs limiar={thr:.4f}."
    elif density is not None:
        interpretation["density_note"] = f"Densidade final={density:.4f}."
    else:
        interpretation["density_note"] = "Densidade não informada."

    # Parada
    if result.get("stopped_by_limit") is True:
        interpretation["stopping_reason"] = "Parada por critério de tempo/gap (solução boa o suficiente no limite configurado)."
    else:
        interpretation["stopping_reason"] = "Parada por término natural (sem limite) ou não informado."

    # Qualidade
    quality = []
    if gap is None:
        quality.append("Gap não informado; avaliar por métricas operacionais e consistência das restrições.")
    elif gap > 0.05:
        quality.append(f"Gap {gap*100:.2f}% (alto): pode haver melhoria considerável com mais tempo.")
    elif gap > 0.01:
        quality.append(f"Gap {gap*100:.2f}% (moderado): aceitável em operação, mas refinável.")
    else:
        quality.append(f"Gap {gap*100:.2f}% (baixo): solução muito boa.")

    interpretation["solution_quality"] = quality

    # Valores do solver
    interpretation["solver_stats"] = {
        "objective_value": result.get("objective_value"),
        "best_bound": result.get("best_bound"),
        "solve_time_s": result.get("solve_time_s") or (result.get("solve_time_ms") / 1000.0 if result.get("solve_time_ms") else None),
        "iterations": result.get("iterations"),
        "num_vars": result.get("num_vars"),
        "num_constraints": result.get("num_constraints"),
    }
    
    # --------------------------------------------------
    # Estrutura do modelo (explicação conceitual)
    # --------------------------------------------------
    model_structure = []

    model_structure.append(
        "Modelo de Programação Inteira Mista (MIP) com variáveis binárias de alocação Y[d,t] "
        "e variáveis de ativação de motorista Z[t]."
    )

    model_structure.append(
        f"Horizonte temporal discretizado em {result.get('num_vars', 'N/A')} variáveis, "
        "representando períodos de 15 minutos."
    )

    if result.get("density") is not None:
        model_structure.append(
            f"Densidade estrutural do modelo: {result['density']:.4f}, "
            "indicando grau de acoplamento entre variáveis e restrições."
        )

    if result.get("num_constraints") is not None:
        model_structure.append(
            f"Total de restrições ativas no modelo: {result['num_constraints']}."
        )

    if result.get("total_active_drivers") is not None:
        model_structure.append(
            f"O modelo decidiu ativar {result['total_active_drivers']} motoristas "
            "para atender a demanda respeitando as restrições legais."
        )

    interpretation["model_structure"] = model_structure
    
    
    # --------------------------------------------------
    # Comportamento do Solver
    # --------------------------------------------------
    solver_behavior = []

    status = result.get("status")
    gap = result.get("gap")
    stopped = result.get("stopped")
    solve_time = result.get("solve_time", 0)
    iterations = result.get("iterations", 0)

    solver_behavior.append(
        "O solver utilizou um algoritmo de Branch-and-Bound com relaxações lineares "
        "e heurísticas internas para exploração do espaço de soluções."
    )

    if status == pywraplp.Solver.OPTIMAL:
        solver_behavior.append(
            "O solver atingiu otimalidade comprovada, com igualdade entre limite primal e dual."
        )
    elif status == pywraplp.Solver.FEASIBLE:
        solver_behavior.append(
            "O solver encontrou uma solução viável, porém sem prova formal de otimalidade."
        )
    elif status == pywraplp.Solver.NOT_SOLVED:
        solver_behavior.append(
            "O solver não conseguiu concluir a resolução dentro dos critérios definidos."
        )

    if gap is not None:
        solver_behavior.append(
            f"O GAP final foi de {gap:.2f}%, indicando a distância entre a melhor solução viável "
            "e o melhor limite dual conhecido."
        )

    if stopped:
        solver_behavior.append(
            "A execução foi interrompida por critério de parada (tempo máximo ou GAP alvo), "
            "antes da convergência completa."
        )

    if solve_time > 0:
        solver_behavior.append(
            f"O tempo total de resolução foi de aproximadamente {solve_time:.1f} segundos, "
            "com {iterations:,} iterações do solver."
        )

    # Interpretação qualitativa
    if gap is not None and gap <= 3:
        solver_behavior.append(
            "O comportamento do solver é considerado satisfatório para problemas "
            "de grande escala com alta densidade combinatória."
        )
    elif gap is not None:
        solver_behavior.append(
            "O GAP residual sugere forte complexidade combinatória e possível necessidade "
            "de decomposição, heurísticas ou relaxações adicionais."
        )

    interpretation["solver_behavior"] = solver_behavior
    

    return interpretation

import numpy as np
import matplotlib.pyplot as plt

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

    # Gini
    horas_acum = np.cumsum(horas)
    horas_acum = np.insert(horas_acum, 0, 0)
    horas_acum_norm = horas_acum / horas_acum[-1]
    proporcao_motoristas = np.linspace(0, 1, len(horas_acum_norm))
    gini = float(1 - 2 * np.trapz(horas_acum_norm, proporcao_motoristas))

    # ------------------------------------------------
    # Layout do painel
    # ------------------------------------------------
    fig = plt.figure(figsize=(12, 7))
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

    ax_ecdf.set_xlabel("Carga total por motorista (h)")
    ax_ecdf.set_ylabel("Proporção acumulada")
    ax_ecdf.set_title("ECDF do Esforço Operacional")
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
    ax_box.set_xticklabels(["Motoristas ativos"])
    ax_box.set_ylabel("Carga total (h)")
    ax_box.set_title("Dispersão do Esforço")
    ax_box.grid(axis="y", linestyle=":", alpha=0.6)

    # ------------------------------------------------
    # 3) Curva de Lorenz
    # ------------------------------------------------
    ax_lorenz.plot(
        proporcao_motoristas,
        horas_acum_norm,
        linewidth=2,
        label="Curva de Lorenz"
    )

    ax_lorenz.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Distribuição equitativa"
    )

    ax_lorenz.set_xlabel("Proporção acumulada de motoristas")
    ax_lorenz.set_ylabel("Proporção acumulada da carga")
    ax_lorenz.set_title(f"Equidade do Esforço Operacional (Gini = {gini:.3f})")
    ax_lorenz.grid(True, linestyle=":", alpha=0.6)
    ax_lorenz.legend()

    # ------------------------------------------------
    # Ajustes finais
    # ------------------------------------------------
    fig.suptitle(
        "Painel de Análise do Esforço Operacional — Motoristas Ativos",
        fontsize=14,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

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



# def interpret_solver_result(result: dict) -> dict:
#     interpretation = {}

#     # -----------------------------
#     # 1. Estado geral do modelo
#     # -----------------------------
#     if result["status"] == pywraplp.Solver.OPTIMAL and result["gap"] <= 0.01:
#         state_msg = "Solução ótima comprovada."
#     elif result["status"] == pywraplp.Solver.OPTIMAL:
#         state_msg = f"Solução quase ótima (gap {result['gap']*100:.2f}%)."
#     elif result["status"] == pywraplp.Solver.FEASIBLE:
#         state_msg = "Solução viável encontrada, mas não ótima."
#     else:
#         state_msg = "Modelo não convergiu para solução viável."

#     interpretation["model_state"] = state_msg

#     # -----------------------------
#     # 2. Qualidade da solução
#     # -----------------------------
#     quality = []
#     if result["gap"] > 0.05:
#         quality.append("Gap elevado — solução pode ser refinada.")
#     elif result["gap"] > 0.01:
#         quality.append("Gap aceitável para planejamento operacional.")
#     else:
#         quality.append("Gap desprezível.")

#     interpretation["solution_quality"] = quality

#     # -----------------------------
#     # 3. Eficiência operacional
#     # -----------------------------
#     total_hours = result["total_assigned_slots"] * 0.25
#     avg_hours = total_hours / max(result["total_active_drivers"], 1)

#     interpretation["operational_metrics"] = {
#         "total_hours": round(total_hours, 2),
#         "avg_hours_per_driver": round(avg_hours, 2),
#         "drivers_used": result["total_active_drivers"]
#     }

#     # -----------------------------
#     # 4. Densidade do modelo
#     # -----------------------------
#     if result["density"] > 0.25:
#         density_msg = "Modelo altamente denso — esperado devido a restrições temporais."
#     else:
#         density_msg = "Modelo esparso."

#     interpretation["model_structure"] = density_msg

#     # -----------------------------
#     # 5. Critério de parada
#     # -----------------------------
#     if result["stopped_by_limit"]:
#         stop_msg = (
#             "O solver foi interrompido por critério de tempo ou gap, "
#             "priorizando desempenho computacional."
#         )
#     else:
#         stop_msg = "O solver explorou completamente o espaço de busca."

#     interpretation["solver_behavior"] = stop_msg

#     return interpretation

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

    fig, ax = plt.subplots(figsize=(11, 4))

    # Barras: capacidade alocada
    ax.bar(
        df["Slot"],
        df["Motoristas"],
        color="steelblue",
        alpha=0.85,
        label="Assigned Drivers (Capacity)"
    )

    # Barras: demanda requerida
    ax.bar(
        df["Slot"],
        df["Demanda"],
        color="salmon",
        alpha=0.85,
        label="Required Drivers (Demand)"
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
        label="Demand Deficit"
    )

    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Drivers")
    ax.set_title("Demand vs. Assigned Capacity (Violations Highlighted)")

    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()





    return fig


def plot_comparative_capacity_analysis(df):
    """
    Painel comparativo:
      (1) Demand vs Capacity com marcação de violação
      (2) Gap (Capacity - Demand) por slot
    """

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True
    )

    # -------------------------------
    # Gráfico 1: Demand vs Capacity
    # -------------------------------
    ax1.bar(
        df["Slot"],
        df["Motoristas"],
        color="steelblue",
        alpha=0.85,
        label="Assigned Drivers (Capacity)"
    )

    ax1.bar(
        df["Slot"],
        df["Demanda"],
        color="salmon",
        alpha=0.85,
        label="Required Drivers (Demand)"
    )

    violated = df[df["Motoristas"] < df["Demanda"]]

    ax1.scatter(
        violated["Slot"],
        violated["Demanda"],
        color="red",
        s=35,
        zorder=5,
        label="Demand Deficit"
    )

    ax1.set_ylabel("Drivers")
    ax1.set_title("Demand vs. Assigned Capacity")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend()

    # -------------------------------
    # Gráfico 2: Gap por slot
    # -------------------------------
    gap = df["Motoristas"] - df["Demanda"]
    colors = gap.apply(lambda x: "red" if x < 0 else "steelblue")

    ax2.bar(
        df["Slot"],
        gap,
        color=colors,
        alpha=0.85
    )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.5)

    ax2.set_xlabel("Time Slot")
    ax2.set_ylabel("Gap (Capacity − Demand)")
    ax2.set_title("Capacity Gap per Slot")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    worst = df.loc[df["Gap"].idxmin()]
    
    ax2.annotate(
        f"Max Deficit = {worst['Gap']}",
        xy=(worst["Slot"], worst["Gap"]),
        xytext=(worst["Slot"], worst["Gap"] - 0.5),
        arrowprops=dict(arrowstyle="->", color="red")
    )


    plt.tight_layout()
    return fig

import matplotlib.pyplot as plt

def plot_carga_motoristas_ativos_vs_inativos(ativos, inativos):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Inativos
    ax1.hist(inativos, bins=1, color="lightgray", edgecolor="black")
    ax1.set_title("Motoristas Inativos")
    ax1.set_xlabel("Slots atribuídos")
    ax1.set_ylabel("Nº de motoristas")
    ax1.set_xticks([0])

    # Ativos
    ax2.hist(ativos, bins=15, color="steelblue", edgecolor="black")
    ax2.set_title("Motoristas Ativos")
    ax2.set_xlabel("Slots atribuídos")

    fig.suptitle("Distribuição de Carga por Motorista")
    plt.tight_layout()

    return fig

def plot_carga_motoristas_ativos(ativos):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(ativos, bins=15, color="steelblue", edgecolor="black")

    ax.set_xlabel("Total de slots atribuídos ao motorista")
    ax.set_ylabel("Número de motoristas")
    ax.set_title("Distribuição da Carga de Trabalho — Motoristas Ativos")

    # Estatísticas-chave
    ax.axvline(ativos.mean(), color="orange", linestyle="--", label=f"Média = {ativos.mean():.1f}")
    ax.axvline(np.median(ativos), color="green", linestyle=":", label=f"Mediana = {np.median(ativos):.1f}")

    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
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
restrictions = [
    {
        "Description": "Coverage Need",
        "Formula": r"$\sum_{j \in \text{valid slot}(i)} X[j] \geq \text{need}[i]$",
        "Detalhes": "The number of workers allocated to periods that meet the valid slots must be sufficient to satisfy the minimum need for the period. \(i\).",
        "Key": "cobertura_necessidade"
    },
    {
        "Description": "Daily Driving Limit",
        "Formula": r"$\sum_{p \in \text{day}} X[p] \leq 36p (1p=15minutes | 36p=9h)$",
        "Detalhes": "It could be 40 periods (10 hours) twice a week.",
        "Key": "limite_diario"
    },
    {
        "Description": "Break 45min After 4.5h Driving",
        "Formula": r"$\text{Pause} \geq 3p (1p=15minutes | 3p=45minutes)$",
        "Detalhes": "After a maximum of 18 periods (4.5 hours), there must be a break of 3 periods (45 minutes).",
        "Key": "pausa_45_minutos"
    },
    # {
    #     "Description": "Split Pause",
    #     "Formula": r"$\text{Pause} | 15 minutes + 30 minutes$",
    #     "Detalhes": "The 3 period break can be done as 15 minutes followed by 30 minutes.",
    #     "Key": "divisao_pausa1530"
    # },
    #     {
    #     "Description": "Split Pause",
    #     "Formula": r"$\text{Pause} | 30 minutes + 15 minutes$",
    #     "Detalhes": "The 3 period break can be done as 30 minutes followed by 15 minutes.",
    #     "Key": "divisao_pausa3015"
    # },
    {
        "Description": "Minimum Daily Rest",
        "Formula": r"$\text{Rest Period} \geq 44p (1p=15minutes |  44p=11h)$",
        "Detalhes": "The driver must rest at least 44 periods (11 hours) every day.",
        "Key": "repouso_diario_minimo"
    },
    {
        "Description": "Weekly Rest",
        # "Formula": r"$\text{Rest Period} \geq 180p (1p=15minutes |  180p=45h) $",
        "Formula": r"$\text{Rest Period} \geq 180p \;\text{(1p = 15 minutes | 180p = 45 h)}$",
        "Detalhes": "The driver must have a rest period of 180 periods (45 hours) every week.",
        "Key": "repouso_semanal"
    },
    {
        "Description": "Rest after 6 days of work",
        "Formula": r"$\text{6 days Work} \Rightarrow \text{Rest for One}$",
        "Detalhes": "Weekly rest must be enjoyed after six consecutive days of work.",
        "Key": "descanso_apos_trabalho"
    },
    {
        "Description": "Weekly Driving Limit",
        "Formula": r"$\sum_{p \in \text{week}} X[p] \leq 224p \;\text{(1p = 15 minutes | 224p = 56 h)}$",
        # "Formula": r"$\sum_{p \in \text{week}} X[p] \leq 224p (1p=15minutes |  224p=56h)$",
        "Detalhes": "Total number of work periods during a week must not exceed 224 periods.",
        "Key": "limite_semanal"
    },
    {
        "Description": "Biweekly Driving Limit",
        "Formula": r"$\sum_{p \in \text{Biweekly }} X[p] \leq 360p (1p=15minutes |  360p=90h)$",
        "Detalhes": "Total number of work periods in two weeks must not exceed 360 periods.",
        "Key": "limite_quinzenal"
    },
    {
        "Description": "Reduced Daily Rest",
        "Formula": r"$\geq 36p \text{ (1p=15minutes |  36p=9h | Max. 3x | 14 days)}$",
        "Detalhes": "Rest may be reduced to 36 periods (9 hours), but not more than 3 times in 14 days.",
        "Key": "repouso_diario_reduzido"
    },
    {
        "Description": "Biweekly Rest",
        "Formula": r"$\geq 96p (1p=15minutes, logo  96p=24h)$",
        "Detalhes": "The driver must have a rest period of 96 periods (24 hours) every two weeks.",
        "Key": "repouso_quinzenal"
    }

]

PROFILES = {

    # =========================
    # 24h — PERFIS BÁSICOS
    # =========================

    "P0_SANITY_24H": {
        "description": "Validação do modelo (cobertura + gráficos, sem regras legais)",
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
        "description": "Operacional diário UE (core)",
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
        "description": "Operacional diário rígido (repouso mínimo)",
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
        "description": "Operacional diário flexível (repouso reduzido permitido)",
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
        "description": "Planejamento semanal (limite semanal)",
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
        "description": "Planejamento semanal rígido (descanso semanal)",
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
        "description": "Planejamento quinzenal (limite quinzenal)",
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
        "description": "Planejamento quinzenal rígido (descansos completos)",
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

    # =========================
    # DEBUG
    # =========================

    "D1_DEBUG_INFEASIBLE": {
        "description": "Diagnóstico de inviabilidade (relaxação progressiva)",
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
    st.subheader("📌 Model f₁ (Assignment)")

    if f1.get("success"):
        st.success("Model f₁ successfully trained!")

        if f1.get("accuracy") is not None:
            st.write(f"**Accuracy:** {f1['accuracy']:.4f}")
        else:
            st.info("Accuracy not available.")

        if "rows_used" in f1:
            st.write(f"**Rows used:** {f1['rows_used']}")

        if f1.get("model_path"):
            st.write(f"**Model saved in:** `{f1['model_path']}`")

    else:
        st.warning("Model f₁ was not trained.")
        st.write(f1.get("message", "Reason not given."))

    # ==========================
    # 🔹 RESULTADOS F2
    # ==========================
    f2 = results.get("f2", {})
    st.subheader("📌 Model f₂ (Neighborhood Ranking)")

    if f2.get("success"):
        st.success("Model f₂ successfully trained!")

        if f2.get("mse") is not None:
            st.write(f"**MSE:** {f2['mse']:.4f}")
        else:
            st.info("MSE not available.")

        if "rows_used" in f2:
            st.write(f"**Rows used:** {f2['rows_used']}")

        if f2.get("model_path"):
            st.write(f"**Model saved in:** `{f2['model_path']}`")

    else:
        st.warning("Model f₂ was not trained.")
        st.write(f2.get("message", "Reason not given."))

# Função para salvar os dados no arquivo
def save_data(data, FILENAME):
  # Carregar dados existentes (se houver)
    existing_data = load_data(FILENAME)
    
    # Se já existem dados salvos
    if existing_data.size > 0:
        # Garantir que ambas as matrizes tenham o mesmo tamanho antes do merge
        max_rows = max(existing_data.shape[0], data.shape[0])
        max_cols = max(existing_data.shape[1], data.shape[1])
        
        # Criar matrizes de tamanho máximo e preencher com zeros
        resized_existing = np.zeros((max_rows, max_cols), dtype=int)
        resized_new = np.zeros((max_rows, max_cols), dtype=int)
        
        # Copiar dados antigos e novos para as matrizes redimensionadas
        resized_existing[:existing_data.shape[0], :existing_data.shape[1]] = existing_data
        resized_new[:data.shape[0], :data.shape[1]] = data
        
        # Atualizar os valores diretamente
        resized_existing[:data.shape[0], :data.shape[1]] = resized_new[:data.shape[0], :data.shape[1]]
        merged_data = resized_existing
        
    else:
        # Se não existem dados antigos, os novos dados são usados diretamente
        merged_data = data
    
    # Salvar a matriz mesclada no arquivo
    with open(FILENAME, 'w') as f:
        json.dump(merged_data.tolist(), f)
       
# Função para carregar os dados do arquivo
def load_data(FILENAME):
    try:
        with open(FILENAME, 'r') as f:
            data = json.load(f)
            return np.array(data)  # Retorna como um array NumPy para facilitar o merge
    except (FileNotFoundError, json.JSONDecodeError):
        return np.array([])  # Retorna um array vazio se o arquivo não existir ou for inválido

# Função para formatar a saída do modelo
def format_lp_output(num_vars, num_restricoes, rhs_values):
    # Parte do objetivo: minimizar a soma de todas as variáveis
    output = f"NumVars:{num_vars}\n\n"
    output += f"NumRestrictions:{num_restricoes}\n\n"
    output += f"Numrhs_values:{len(rhs_values)}\n\n"
    
    # Cabeçalho e função objetivo
    output += f"MODEL:\n\n"
    output += " [_1] MIN= " + " + ".join(f"X_{i+1}" for i in range(num_vars)) + ";\n\n"

    # Restrições: usa o mesmo conjunto de variáveis para cada restrição
    restricao = " + ".join(f"X_{i+1}" for i in range(num_vars))
    
    for j in range(num_vars):
        # Define o valor do RHS para cada restrição
        rhs_value = rhs_values[j] if j < len(rhs_values) else 0
        output += f" [_{j+2}] {restricao} >= {rhs_value};\n\n"

    # Variáveis inteiras
    output += "\n" + " ".join(f"@GIN(X_{i+1});" for i in range(num_vars)) + "\n"
    output += "END\n"

    return output

# Função para determinar o tipo de modelo
def tipo_modelo(solver):
    return "Linear or Integer Model"

# Função de cálculo da densidade
def calculate_density(matrix):
    # Check if matrix is empty or None
    if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
        return 0
    # Calculate non-zero count
    non_zero_count = sum(sum(1 for elem in row if elem != 0) for row in matrix)
    # Calculate total count
    total_count = len(matrix) * len(matrix[0])
    # Calculate density
    density = non_zero_count / total_count if total_count > 0 else 0
    return density

# Função para exibir o sistema linear no formato desejado
def display_system(constraints_coefficients, need):
    system = ""
    for i, row in enumerate(constraints_coefficients):
        equation = " + ".join([f"{coef}*X{j+1}" for j, coef in enumerate(row) if coef != 0])
        equation += f" = {need[i]}"
        system += equation + "\n"
    st.text(system)

def rebuild_allocation_from_schedule(
    workers_schedule: list[int],
    need: list[int],
    limit_workers: int
) -> np.ndarray:
    """
    Reconstrói uma matriz de alocação (period x worker)
    a partir do workers_schedule devolvido pelo solver exact.
    """
    num_periods = len(workers_schedule)
    allocation = np.zeros((num_periods, limit_workers), dtype=int)

    for p, w in enumerate(workers_schedule):
        w = min(w, limit_workers)
        allocation[p, :w] = 1

    return allocation

def normalize_solver_outputs(
    demanda: List[int],
    workers_schedule: Any,
    matrix_allocation: Any
) -> Tuple[Optional[List[int]], Optional[np.ndarray]]:
    """
    Normaliza os outputs do solver para uso seguro na UI e KPIs.

    Regras:
    - workers_schedule → List[int] ou None
    - matrix_allocation → np.ndarray ou None
    - nunca lança exceção
    """

    # --------------------------------------------------
    # 1) Normalizar matrix_allocation
    # --------------------------------------------------
    # if matrix_allocation is not None:
    #     try:
    #         matrix_allocation = np.asarray(matrix_allocation, dtype=int)
    #         if matrix_allocation.ndim != 2:
    #             matrix_allocation = None
    #     except Exception:
    #         matrix_allocation = None
    
    # --------------------------------------------------
    # 1) Normalizar matrix_allocation
    # --------------------------------------------------
    if matrix_allocation is not None:
        try:
            matrix_allocation = np.asarray(matrix_allocation, dtype=int)
            if matrix_allocation.ndim != 2:
                matrix_allocation = None
        except Exception:
            matrix_allocation = None

    # 🔥 NOVO: tentar reconstruir matrix_allocation
    if matrix_allocation is None and workers_schedule is not None:
        try:
            # fallback: matriz diagonal simples (slot x 1 worker agregado)
            matrix_allocation = np.array(workers_schedule, dtype=int).reshape(-1, 1)
        except Exception:
            matrix_allocation = None
    

    # --------------------------------------------------
    # 2) Normalizar workers_schedule
    # --------------------------------------------------
    ws: Optional[List[int]] = None

    # Caso ideal: já é lista
    # if isinstance(workers_schedule, list):
    #     ws = [int(x) for x in workers_schedule]
    
    # Derivar SEMPRE da matriz, se possível
    if matrix_allocation is not None:
        try:
            ws = list(np.sum(matrix_allocation, axis=1).astype(int))
        except Exception:
            ws = None    

    # Caso: numpy array 1D
    elif isinstance(workers_schedule, np.ndarray):
        if workers_schedule.ndim == 1:
            ws = [int(x) for x in workers_schedule.tolist()]

    # Caso: int ou float → inválido para KPIs
    elif isinstance(workers_schedule, (int, float)):
        ws = None

    # Caso: None → tentar derivar da matriz
    if ws is None and matrix_allocation is not None:
        try:
            ws = list(np.sum(matrix_allocation, axis=1).astype(int))
        except Exception:
            ws = None

    # --------------------------------------------------
    # 3) Validar tamanho vs demanda
    # --------------------------------------------------
    if ws is not None and len(ws) != len(demanda):
        ws = None  # evita KPIs inconsistentes

    return ws, matrix_allocation

# ============================================================
# Função controladora para os três modos: Exact, Heuristic, LNS
# ============================================================
def run_solver_with_mode(
    mode,
    need,
    variable_type,
    constraints_coefficients,
    selected_restrictions,
    solver_param_type,
    densidade_aceitavel,     
    limit_workers,
    limit_iteration,
    limit_level_relaxation,
    max_demands_per_driver,
    tolerance_demands,
    penalty,
    swap_rows=None, 
    multiply_row=None, 
    add_multiple_rows=None,
    radio_selection_object=None 
):

    # 1) Sempre gerar a heurística inicial
    initial_allocation = greedy_initial_allocation(
        need=need,
        limit_workers=limit_workers,
        max_demands_per_driver=max_demands_per_driver,
        assignment_scorer_fn=assignment_scorer if assignment_scorer is not None else None
    )
    
    
    # ======================================================
    # 🔵 FUNÇÃO AUXILIAR: Retorno padronizado para os 3 modos
    # ======================================================
    def make_return(
        solver=None,
        status="NOT_SOLVED",
        total_active_drivers=0,
        total_assigned_slots=0,        
        workers_schedule=None,
        constraints_coeffs=None,
        initial_density=None,
        final_density=None,
        statistics=None,
        msg=None,
        iterations=None,
        allocation=None,
        logs=None,
    ):
        # Garante estruturas mínimas
        if statistics is None:
            statistics = []
        if msg is None:
            # msg = []
            msg: list[str] = []   # ✅ garante contrato estável
        if iterations is None:
            iterations = []
        if logs is None:
            logs = {"stdout": "", "stderr": ""}

        return (
            solver,             # 1
            status,             # 2
            total_active_drivers, # 3
            total_assigned_slots, # 4            
            workers_schedule,   # 5
            tasks_schedule,
            driving_hours_per_driver,
            constraints_coeffs, # 6
            initial_density,    # 7
            final_density,      # 8
            statistics,         # 9
            msg,                # 10
            iterations,         #11
            allocation,         #12
            logs                #13
        )
    
    
    
    # ======================================================
    # MODO EXATO
    # ======================================================
    if mode == "Exact":
        (
            solver,
            status,
            total_active_drivers, 
            total_assigned_slots,             
            workers_schedule,
            tasks_schedule,
            driving_hours_per_driver,
            constraints_coefficients_out,
            initial_density,
            final_density,
            statistics_result,
            msg,
            iterations_data,
            matrix_allocation,
            solver_logs,
        ) = solve_shift_schedule(
            solver_param_type,
            need,
            variable_type,
            constraints_coefficients,
            selected_restrictions,
            swap_rows,
            multiply_row,
            add_multiple_rows,
            densidade_aceitavel,
            limit_workers,
            limit_iteration,
            limit_level_relaxation,
            max_demands_per_driver,
            tolerance_demands,
            penalty,
            initial_allocation=None,
            fixed_assignments=None,
            radio_selection_object=radio_selection_object,
            mode="Exact"
        )
        
        # ==========================================================
        # NORMALIZAÇÃO CANÔNICA (🔥 ESSENCIAL 🔥)
        # ==========================================================
        # workers_schedule, matrix_allocation = normalize_solver_outputs(
        #     demanda=need,
        #     workers_schedule=workers_schedule,
        #     matrix_allocation=matrix_allocation
        # )
        
        # ==========================================================
        # NORMALIZAÇÃO CANÔNICA (MODO EXACT)
        # ==========================================================

        # 1️⃣ Se o solver não construiu matriz, reconstruir
        if matrix_allocation is None and workers_schedule is not None:
            matrix_allocation = rebuild_allocation_from_schedule(
                workers_schedule=workers_schedule,
                need=need,
                limit_workers=limit_workers
            )

        # 2️⃣ Normalizar formatos
        workers_schedule, matrix_allocation = normalize_solver_outputs(
            demanda=need,
            workers_schedule=workers_schedule,
            matrix_allocation=matrix_allocation
        )

        # 3️⃣ Última linha de defesa (NUNCA retornar None)
        if matrix_allocation is None:
            matrix_allocation = greedy_initial_allocation(
                need=need,
                limit_workers=limit_workers,
                max_demands_per_driver=max_demands_per_driver
            )

        return make_return(
            solver,
            status,
            total_active_drivers,
            total_assigned_slots,
            workers_schedule,
            constraints_coefficients_out,
            initial_density,
            final_density,
            statistics_result,
            msg,
            iterations_data,
            matrix_allocation,
            solver_logs
        )
        
    # ======================================================
    # MODO HEURÍSTICO
    # ======================================================
   
    if mode == "Heuristic":
        matrix_allocation = initial_allocation
        # total_workers = int(matrix_allocation.sum())
        
        # total de slots alocados (esforço total)
        total_assigned_slots = int(matrix_allocation.sum())

        # total de motoristas ativados (decisão)
        total_active_drivers = int(
            np.sum(np.any(matrix_allocation > 0, axis=0))
        )
        
        workers_schedule = list(np.sum(matrix_allocation, axis=1))
        
        statistics_result = [
            "Model State: HEURISTIC",
            f"Total active drivers: {total_active_drivers}",
            f"Total assigned slots: {total_assigned_slots}",
            "Model Type: Heuristic (greedy initial allocation)"
        ]

        # Para manter a mesma assinatura da função (12 retornos),
        # mas sem erro de variável não definida:
        # solver_logs =  {"stdout": "", "stderr": ""}   # nenhum log específico no modo heurístico

        # ==========================================================
        # NORMALIZAÇÃO CANÔNICA (🔥 ESSENCIAL 🔥)
        # ==========================================================
        workers_schedule, matrix_allocation = normalize_solver_outputs(
            demanda=need,
            workers_schedule=workers_schedule,
            matrix_allocation=matrix_allocation
        )


        return make_return(
            solver=None,
            status="HEURISTIC",
            total_active_drivers=total_active_drivers,
            total_assigned_slots=total_assigned_slots,
            workers_schedule=workers_schedule,
            constraints_coeffs=constraints_coefficients,
            initial_density=None,
            final_density=None,
            statistics=statistics_result,
            msg=[],
            iterations=[],
            allocation=matrix_allocation,
            logs={"stdout": "Heuristic mode executed.", "stderr": ""}
        )   

    # ======================================================
    # MODO LNS (MATHEURÍSTICO)
    # ======================================================
   
    if mode == "LNS":
        matrix_allocation = None
        best_solution, info = run_lns(
            initial_solution=initial_allocation,
            need=need,
            variable_type=variable_type,
            constraints_coefficients=constraints_coefficients,
            selected_restrictions=selected_restrictions,
            solver_param_type=solver_param_type,
            # total_active_drivers=total_active_drivers, # 3
            # total_assigned_slots=total_assigned_slots, # 4            
            limit_workers=limit_workers,
            limit_iteration=limit_iteration,
            limit_level_relaxation=limit_level_relaxation,
            max_demands_per_driver=max_demands_per_driver,
            tolerance_demands=tolerance_demands,
            penalty=penalty,
            max_lns_iterations=5,
            solve_fn=solve_shift_schedule,
            neighborhood_scorer_fn=neighborhood_scorer if neighborhood_scorer is not None else None
        )
        
        # Fallback de segurança
        if best_solution is None or not isinstance(best_solution, np.ndarray):
            best_solution = initial_allocation.copy()


        # ✔️ esforço total
        total_assigned_slots = int(best_solution.sum())

        # ✔️ decisão real (objetivo do modelo)
        total_active_drivers = int(
            np.sum(np.any(best_solution > 0, axis=0))
        )

        workers_schedule = list(np.sum(best_solution, axis=1))

        # total_workers = int(best_solution.sum())
        # workers_schedule = list(np.sum(best_solution, axis=1))

        statistics_result = [
            "Model State: LNS",
            f"Total active drivers: {total_active_drivers}",
            f"Total assigned slots: {total_assigned_slots}",
            "Model Type: Matheuristic (LNS + MILP)"
        ]

        # Para manter a mesma assinatura da função (12 retornos),
        # mas sem erro de variável não definida:
        # solver_logs = {"stdout": "", "stderr": ""}  # nenhum log específico no modo heurístico        

        # ==========================================================
        # NORMALIZAÇÃO CANÔNICA (🔥 ESSENCIAL 🔥)
        # ==========================================================
        workers_schedule, matrix_allocation = normalize_solver_outputs(
            demanda=need,
            workers_schedule=workers_schedule,
            matrix_allocation=best_solution
        )

        # 🔒 ÚLTIMO FALLBACK — NUNCA RETORNAR None
        if matrix_allocation is None:
            matrix_allocation = initial_allocation.copy()
    
        return make_return(
            solver=None,
            status="LNS",
            total_active_drivers=total_active_drivers,
            total_assigned_slots=total_assigned_slots,
            workers_schedule=workers_schedule,
            constraints_coeffs=constraints_coefficients,
            initial_density=None,
            final_density=None,
            statistics=statistics_result,
            msg=[],
            iterations=info.get("history", []),
            allocation=matrix_allocation,
            logs={"stdout": "LNS executed successfully.", "stderr": ""}
        )

# Função para relaxar restrições dinamicamente em caso de conflitos
def relax_restrictions(solver, constraints, relaxation_level):
    for constraint in constraints:
        # Ajusta o limite superior ou inferior das restrições para relaxar o problema
        if constraint.Lb() is not None:
            constraint.SetLb(constraint.Lb() - relaxation_level)
        if constraint.Ub() is not None:
            constraint.SetUb(constraint.Ub() + relaxation_level)

def relax_objective(solver, Y, num_periods, limit_workers, penalty, max_penalty, relaxation_level, need, tolerance_demands,  max_demands_per_driver,):
    """
    Relaxar a função objetivo, diminuindo a penalização de demandas conforme o relaxamento aumenta.
    """
    # Ajustar a penalização progressivamente
    relaxed_penalty = max(0, penalty - relaxation_level)  # Diminui a penalização até zero
    
    # Construir a função objetivo com penalização relaxada
    objective = solver.Minimize(
        solver.Sum(Y[d, t] for d in range(num_periods) for t in range(limit_workers))  # Minimizar motoristas
        + relaxed_penalty * solver.Sum(
            (need[d] * tolerance_demands - solver.Sum(Y[d, t] * max_demands_per_driver for t in range(limit_workers))) 
            for d in range(num_periods)
        )
    )
    
    return objective

def get_solver_status_description(status_code):
    status_mapping = {
        0: "OPTIMAL",
        1: "FEASIBLE",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "ABNORMAL",
        5: "MODEL_INVALID",
        6: "NOT_SOLVED"
    }
    return status_mapping.get(status_code, "UNKNOWN_STATUS")

# Função de otimização (baseada no código anterior)
def solve_shift_schedule(
    solver_param_type, 
    need, 
    variable_type, 
    constraints_coefficients, 
    selected_restrictions, 
    swap_rows=None, 
    multiply_row=None, 
    add_multiple_rows=None, 
    densidade_aceitavel=None, 
    limit_workers=0,
    limit_iteration=0,
    limit_level_relaxation=0, 
    cap_tasks_per_driver_per_slot=1, # (ex.: 3 tarefas/15min) 
    tolerance_demands=0.01, 
    penalty = 0.01,
    initial_allocation=None,      # <<< NOVO
    fixed_assignments=None,       # <<< NOVO
    radio_selection_object=None,  # <<< NOVO
    mode="Exact"
):
    
    constraints = []
    msg: list[str] = []
    num_periods = len(need)
    Y = {}

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    if mode == "Exact" and radio_selection_object is None:
        raise ValueError("radio_selection_object must be provided for Exact mode")

    with contextlib.redirect_stdout(stdout_buffer):
        with contextlib.redirect_stderr(stderr_buffer):    
            
            # Criar o solver
            solver = pywraplp.Solver.CreateSolver(solver_param_type)  # Usar 'GLOP' para LP ou 'SCIP' para MIP
            solver.EnableOutput()
            
            # ==============================
            # ⛔ PONTO DE PARADA CONTROLADO
            # ==============================
            TIME_LIMIT_SEC = 300        # 120 - ou vindo do Streamlit
            GAP_REL = 0.10              # 0.05 - 5%

            solver.SetSolverSpecificParametersAsString(f"""
            limits/time = {TIME_LIMIT_SEC}
            limits/gap = {GAP_REL}
            display/verblevel >= 1
            """)
            
            # Level	Comportamento
            # 0	🔇 totalmente silencioso
            # 1	logs mínimos (status final)
            # 2	heurísticas + bounds (✅ ideal)
            # 3	logs detalhados
            # 4	debug pesado (❌ evite)            
            
            Z = {}
            for t in range(limit_workers):
                Z[t] = solver.BoolVar(f"Z_{t}")

            U = {}
            for d in range(num_periods):
                U[d] = solver.NumVar(0, need[d], f"U[{d}]")         
                
            # -------------------------------
            #  Quantas tarefas o motorista atende no slot (capacidade contínua)
            # -------------------------------
            X = {}
            for d in range(num_periods):
                for t in range(limit_workers):
                    # X[d, t] = solver.NumVar(0, solver.infinity(), f"X[{d},{t}]")
                    X[d, t] = solver.IntVar(0, cap_tasks_per_driver_per_slot, f"X[{d},{t}]")
                    
            for d in range(num_periods):
                for t in range(limit_workers):
                    if variable_type == "Continuous":
                        Y[d, t] = solver.NumVar(0, 1, f'Y[{d}, {t}]')  # Variável contínua
                    elif variable_type == "Binary":
                        Y[d, t] = solver.BoolVar(f'Y[{d}, {t}]')  # Variável binária
                    elif variable_type == "Integer":
                        Y[d, t] = solver.IntVar(0, 1, f'Y[{d}, {t}]')  # Variável inteira
                    else:
                        raise ValueError("Invalid variable type for Y[d, t].")

            # 1) Capacidade
            for d in range(num_periods):
                for t in range(limit_workers):
                    solver.Add(X[d, t] <= cap_tasks_per_driver_per_slot * Y[d, t])

            # Link working variables to driver activation
            for d in range(num_periods):
                for t in range(limit_workers):
                    solver.Add(Y[d, t] <= Z[t])
                    
            # 2) P"rodução mínima se motorista está ativo
            # for d in range(num_periods):
            #     for t in range(limit_workers):
            #         solver.Add(X[d, t] >= Y[d, t])


            # 3) Ativação correta
            for t in range(limit_workers):
                solver.Add(solver.Sum(Y[d, t] for d in range(num_periods)) <= num_periods * Z[t])


            # ----------------------------------------------------
            # 1) CARGA POR MOTORISTA (L) e MÁXIMA CARGA (Lmax)
            # ----------------------------------------------------
            L = {}
            for t in range(limit_workers):
                # L[t] = solver.NumVar(0, cap_tasks_per_driver_per_slot * num_periods, f"L[{t}]")
                # solver.Add(L[t] == solver.Sum(Y[d, t] for d in range(num_periods)))
                # solver.Add(L[t] <= num_periods * Z[t])  # ajuste o bound se usar Y
                L[t] = solver.NumVar(0, cap_tasks_per_driver_per_slot * num_periods, f"L[{t}]")
                solver.Add(L[t] == solver.Sum(X[d, t] for d in range(num_periods)))
                solver.Add(L[t] <= cap_tasks_per_driver_per_slot * num_periods * Z[t]) 


            Lmax = solver.NumVar(0, cap_tasks_per_driver_per_slot * num_periods, "Lmax")
            for t in range(limit_workers):
                solver.Add(L[t] <= Lmax)


            S = {}
            E = {}
            SHIFT_MIN = 36   # 9h  (mínimo de turno)
            SHIFT_MAX = 52   # 13h (turno realista UE)
            

            for t in range(limit_workers):
                for d in range(num_periods):
                    S[d, t] = solver.BoolVar(f"S[{d},{t}]")
                    E[d, t] = solver.BoolVar(f"E[{d},{t}]")

                # Um único início e um único fim se ativo
                solver.Add(solver.Sum(S[d, t] for d in range(num_periods)) == Z[t])
                solver.Add(solver.Sum(E[d, t] for d in range(num_periods)) == Z[t])

                # Duração do turno (presença)
                solver.Add(
                    solver.Sum(Y[d, t] for d in range(num_periods)) >= SHIFT_MIN * Z[t]
                )
                solver.Add(
                    solver.Sum(Y[d, t] for d in range(num_periods)) <= SHIFT_MAX * Z[t]
                )

                # Condição inicial
                solver.Add(E[0, t] == 0)
                solver.Add(Y[0, t] == S[0, t])

                # Continuidade (bloco único)
                for d in range(1, num_periods):
                    solver.Add(
                        Y[d, t] - Y[d - 1, t] == S[d, t] - E[d, t]
                    )
                


            # ----------------------------------------------------
            # 2) BALANCEAMENTO (Over / Dev) - AGORA L EXISTE
            # ----------------------------------------------------
            SOFT_CAP = 32 * cap_tasks_per_driver_per_slot

            Over = {}
            Dev = {}
            for t in range(limit_workers):
                Over[t] = solver.NumVar(0, cap_tasks_per_driver_per_slot * num_periods, f"Over[{t}]")
                Dev[t]  = solver.NumVar(0, cap_tasks_per_driver_per_slot * num_periods, f"Dev[{t}]")

                solver.Add(Over[t] >= L[t] - SOFT_CAP)
                solver.Add(Over[t] >= 0)
                solver.Add(Over[t] <= cap_tasks_per_driver_per_slot * num_periods * Z[t])

                # (opcional) "abaixo do soft cap" como desvio também
                solver.Add(Dev[t] >= SOFT_CAP - L[t])
                solver.Add(Dev[t] >= 0)
                solver.Add(Dev[t] <= cap_tasks_per_driver_per_slot * num_periods * Z[t])

                    

            if (radio_selection_object == "Maximize Demand Response"):
                # Função objetivo: maximizar o atendimento de demanda considerando as tolerâncias e limites
                solver.Maximize(
                    solver.Sum(X[d, t] * need[d] for d in range(num_periods) for t in range(limit_workers))  # Maximizar o atendimento da demanda
                    - penalty * solver.Sum(
                        (need[d] - solver.Sum(U[d] for d in range(num_periods)))
                        for d in range(num_periods)  # Penalização para garantir que a demanda atendida seja dentro da tolerância
                    )
                )

                # Restrição para distribuir o atendimento de forma proporcional:    
                for d in range(num_periods):
                    solver.Add(solver.Sum(Y[d, t] for t in range(limit_workers)) <= need[d])  # Garantir que a demanda não seja superada por período

            elif (radio_selection_object == "Minimize Total Number of Drivers"):

                BIG_UNMET = 1_000_000
                BIG_Z     = 10_000
                REWARD_X = 1.0

                # -------------------------
                # FASE 1: minimizar U e Z
                # -------------------------
                # FASE 1: prioridade absoluta = cobrir demanda (U)
                # depois = minimizar motoristas (Z)
                solver.Minimize(
                    BIG_UNMET * solver.Sum(U[d] for d in range(num_periods))
                + BIG_Z     * solver.Sum(Z[t] for t in range(limit_workers))
                - REWARD_X  * solver.Sum(X[d, t] for d in range(num_periods) for t in range(limit_workers))
                )
                
            #01
            # need[d] = tarefas
            # X[d,t] = tarefas atendidas
            # U[d] = tarefas não atendidas

            if selected_restrictions["cobertura_necessidade"]: 
                # -------------------------------
                # DEMANDA OBRIGATÓRIA (HARD)
                # -------------------------------
                for d in range(num_periods):
                    solver.Add(
                        solver.Sum(X[d, t] for t in range(limit_workers)) + U[d]  == need[d]
                    )
                    
            if fixed_assignments:
                for (d_fix, t_fix, val) in fixed_assignments:
                    # SCIP requer variáveis booleanas explicitamente fixadas assim:
                    if val == 1:
                        Y[d_fix, t_fix].SetBounds(1, 1)
                    else:
                        Y[d_fix, t_fix].SetBounds(0, 0)          

            #02
            if selected_restrictions.get("pausa_45_minutos", False):

                max_continuous_work = 18   # 4h30 = 18 slots
                pause_duration = 3         # 45 min = 3 slots
                window = max_continuous_work + pause_duration  # 21

                for t in range(limit_workers):
                    for start in range(num_periods - window + 1):
                        constraints.append(
                            solver.Add(solver.Sum(X[start + p, t] for p in range(window)) <= max_continuous_work)
                        )

            #03
            if selected_restrictions["limite_diario"]: 
                max_daily_driving = 36  # 9 horas (36 períodos de 15 minutos)
                periods_per_day = 96
                
                for day in range(num_periods // periods_per_day):  # 96 períodos de 15 minutos em um dia
                    
                    # Cria a expressão para a soma de X[j] nos períodos válidos
                    day_start = day * periods_per_day  # Início do dia em termos de períodos
                    day_end = (day + 1) * periods_per_day  # Fim do dia em termos de períodos (não inclusivo)
                    
                    # ✅ Limite diário POR MOTORISTA
                    for t in range(limit_workers):
                        constraint_expr = solver.Sum(X[d, t] for d in range(day_start, day_end))                    
                    
                        # Adiciona a restrição de limite diário, mas com mais flexibilidade se o repouso for reduzido
                        constraint = solver.Add(constraint_expr <= max_daily_driving)
                        constraints.append(constraint)
        
            if selected_restrictions["repouso_diario_minimo"]:
                periods_per_day = 96

                max_work_normal = 52   # 96 - 44 (11h repouso)
                max_work_reduced = 60  # 96 - 36 (9h repouso)

                reduced_rest_days = 3  # no máximo 3 repousos reduzidos

                for t in range(limit_workers):

                    reduced_rest_vars = []

                    for day in range(num_periods // periods_per_day):
                        reduced_rest = solver.BoolVar(f"reduced_rest_d{day}_t{t}")
                        reduced_rest_vars.append(reduced_rest)

                        day_start = day * periods_per_day
                        day_end = day_start + periods_per_day

                        daily_work = solver.Sum(X[d, t] for d in range(day_start, day_end))

                        constraints.append(
                            solver.Add(
                                daily_work
                                <= max_work_reduced * reduced_rest
                                + max_work_normal * (1 - reduced_rest)
                            )
                        )

                    # no máximo 3 dias com repouso diário reduzido
                    constraints.append(
                        solver.Add(solver.Sum(reduced_rest_vars) <= reduced_rest_days)
                    )


            if selected_restrictions["limite_quinzenal"]:
                periods_per_day = 96
                periods_per_week = periods_per_day * 7
                periods_per_fortnight = periods_per_week * 2

                max_fortnight_driving = 360  # 90h = 360 períodos de 15 min

                for fortnight in range(num_periods // periods_per_fortnight):
                    fortnight_start = fortnight * periods_per_fortnight
                    fortnight_end = (fortnight + 1) * periods_per_fortnight

                    for t in range(limit_workers):
                        constraint_expr = solver.Sum(
                            Y[d, t] for d in range(fortnight_start, fortnight_end)
                        )

                        constraints.append(
                            solver.Add(constraint_expr <= max_fortnight_driving)
                        )


            #6
            if selected_restrictions["repouso_diario_reduzido"]:

                periods_per_day = 96
                periods_per_week = periods_per_day * 7

                min_weekly_rest = 96        # 24h = 96 períodos
                max_reduced_weeks = 1       # no máximo 1 repouso semanal reduzido (em 2 semanas)

                for t in range(limit_workers):

                    reduced_week_vars = []

                    for week in range(num_periods // periods_per_week):
                        reduced_week = solver.BoolVar(f"reduced_week_rest_w{week}_t{t}")
                        reduced_week_vars.append(reduced_week)

                        week_start = week * periods_per_week
                        week_end = week_start + periods_per_week

                        weekly_work = solver.Sum(
                            Y[d, t] for d in range(week_start, week_end)
                        )

                        # trabalho máximo permitido na semana
                        constraints.append(
                            solver.Add(
                                weekly_work
                                <= periods_per_week - min_weekly_rest * (1 - reduced_week)
                            )
                        )

                    # no máximo 1 semana com repouso semanal reduzido
                    constraints.append(
                        solver.Add(solver.Sum(reduced_week_vars) <= max_reduced_weeks)
                    )
            
            if selected_restrictions["limite_semanal"]:
                periods_per_day = 96
                periods_per_week = 96 * 7

                max_daily_normal = 36   # 9h
                max_daily_extended = 40 # 10h
                max_weekly_driving = 224  # 56h

                extended_days_limit = 2

                for week in range(num_periods // periods_per_week):
                    week_start = week * periods_per_week
                    week_end = (week + 1) * periods_per_week

                    for t in range(limit_workers):

                        # ---- Exceção de 10h (no máximo 2 dias na semana) ----
                        extended_day_vars = []

                        for day in range(7):
                            day_start = week_start + day * periods_per_day
                            day_end = day_start + periods_per_day

                            extended = solver.BoolVar(f"ext_week{week}_day{day}_t{t}")
                            extended_day_vars.append(extended)

                            daily_expr = solver.Sum(X[d, t] for d in range(day_start, day_end))

                            constraints.append(
                                solver.Add(
                                    daily_expr <= max_daily_extended * extended
                                                + max_daily_normal * (1 - extended)
                                )
                            )

                        # no máximo 2 dias com 10h
                        constraints.append(
                            solver.Add(solver.Sum(extended_day_vars) <= extended_days_limit)
                        )

                        # ---- Limite semanal absoluto (56h) ----
                        weekly_expr = solver.Sum(
                            Y[d, t] for d in range(week_start, week_end)
                        )

                        constraints.append(
                            solver.Add(weekly_expr <= max_weekly_driving)
                        )
            
            if selected_restrictions["descanso_apos_trabalho"]:

                periods_per_day = 96
                periods_per_week = periods_per_day * 7
                min_weekly_rest = 96  # 24h de repouso semanal reduzido (mínimo legal)

                for t in range(limit_workers):

                    for week in range(num_periods // periods_per_week):
                        week_start = week * periods_per_week
                        week_end = week_start + periods_per_week

                        # total de períodos trabalhados na semana
                        weekly_work = solver.Sum(
                            Y[d, t] for d in range(week_start, week_end)
                        )

                        # deve existir pelo menos 24h (96 períodos) sem trabalho
                        constraints.append(
                            solver.Add(weekly_work <= periods_per_week - min_weekly_rest)
                        )
            
            # Aplicar operações elementares, se selecionadas
            if swap_rows is not None:
                # Exibir o sistema linear antes da troca de linhas
                row1, row2 = swap_rows
                if (0 <= row1 < len(constraints_coefficients)) and (0 <= row2 < len(constraints_coefficients)):
                    # Troca de linhas na matriz de coeficientes
                    constraints_coefficients[row1], constraints_coefficients[row2] = constraints_coefficients[row2], constraints_coefficients[row1]
                    # Troca correspondente no vetor de resultados
                    need[row1], need[row2] = need[row2], need[row1]
                    # Atualização de restrições no solver
                    constraints[row1], constraints[row2] = constraints[row2], constraints[row1]
                    print(f"Rows {row1} e {row2} exchanged successfully!")

            # A multiplicação de uma linha por uma constante pode ser útil para simplificar valores,
            # mas deve ser controlada para evitar o aumento do "fill-in".
            if multiply_row is not None:
                row, constant = multiply_row
                
                # Verificar se o índice da linha está dentro dos limites da matriz
                if 0 <= row < len(constraints_coefficients):
                    
                    # Evitar multiplicação por 0
                    if constant != 0:
                        
                        # Multiplicar a linha pelos coeficientes, levando em consideração a estrutura de 3D
                        for d in range(len(constraints_coefficients[row])):
                            if isinstance(constraints_coefficients[row][d], (list, np.ndarray)):  # Verifica se é uma estrutura iterável
                                # Se for uma lista ou array, iterar sobre t
                                for t in range(len(constraints_coefficients[row][d])):
                                    constraints_coefficients[row][d][t] *= constant
                            else:
                                # Se for um valor escalar, apenas multiplica diretamente
                                constraints_coefficients[row][d] *= constant

                        # Atualizar a expressão de restrição correspondente no solver
                        new_expr = solver.Sum(
                            # Agora verificamos se constraints_coefficients[row][d] é uma estrutura iterável
                            (constraints_coefficients[row][d][t] if isinstance(constraints_coefficients[row][d], (list, np.ndarray)) 
                            else constraints_coefficients[row][d]) * Y[d, t]
                            for d in range(num_periods)
                            for t in range(limit_workers)
                        )
                        
                        # Atualizar a restrição com a multiplicação da constante
                        constraints[row] = solver.Add(new_expr >= constraints[row].lb() * constant)
                        
                        # Mensagem de sucesso
                        print(f"Row {row} multiplied by {constant} successfully!")
                    else:
                        print(f"Skipping multiplication for row {row} as constant is zero.")
                else:
                    print(f"Invalid row index {row} for constraints_coefficients.")

            # Multiplicação automática de linhas
            if add_multiple_rows is not None:
                row1, row2, multiple = add_multiple_rows

                # Exibir informações de depuração sobre a estrutura da matriz
                print(f"Structure of constraints_coefficients: {type(constraints_coefficients)}, Shape: {np.array(constraints_coefficients).shape}")

                # Verifique se os índices das linhas são válidos
                if 0 <= row1 < len(constraints_coefficients) and 0 <= row2 < len(constraints_coefficients):
                    print(f"Applying add_multiple_rows operation: row1={row1}, row2={row2}, multiple={multiple}")

                    # Verificar se constraints_coefficients[row1] e constraints_coefficients[row2] são listas ou arrays
                    if isinstance(constraints_coefficients[row1], (list, np.ndarray)) and isinstance(constraints_coefficients[row2], (list, np.ndarray)):
                        
                        # Criar os novos valores para a linha row2, adicionando o múltiplo de row1
                        new_row_values = [
                            constraints_coefficients[row2][j] + multiple * constraints_coefficients[row1][j]
                            for j in range(len(constraints_coefficients[row2]))
                        ]

                        # Verificar se a linha resultante tem valores não nulos (com tolerância para zero)
                        if any(abs(value) > 1e-6 for value in new_row_values):
                            constraints_coefficients[row2] = new_row_values
                            need[row2] = need[row2] + multiple * need[row1]

                            # Multiplicar a linha por -1 se houver coeficientes negativos
                            if any(value < 0 for value in constraints_coefficients[row2]):
                                constraints_coefficients[row2] = [-value for value in constraints_coefficients[row2]]
                                need[row2] = -need[row2]

                            # Atualizar a restrição no solver com os novos valores
                            try:
                                new_expr = solver.Sum(
                                    constraints_coefficients[row2][j] * Y[d, t]
                                    for j in range(len(constraints_coefficients[row2]))
                                    for d in range(num_periods)
                                    for t in range(limit_workers)
                                    if (d, t) in Y  # Verificar se a chave existe no dicionário
                                )
                                constraints[row2] = solver.Add(new_expr >= need[row2])
                                print(f"Row {row2} updated successfully: {constraints_coefficients[row2]}")
                            except KeyError as e:
                                print(f"KeyError: {e}. Invalid key in Y or mismatch in constraints_coefficients.")
                            except Exception as e:
                                print(f"Unexpected error: {e}")
                        else:
                            print(f"The operation resulted in a null line for row {row2}, which was avoided.")
                    else:
                        print(f"Error: Unexpected structure for rows {row1} or {row2}. Ensure constraints_coefficients is 2D.")
                else:
                    print(f"Invalid indices for add_multiple_rows operation: row1={row1}, row2={row2}")

            # Dentro da sua função solve_shift_schedule
            final_density = calculate_density(constraints_coefficients)
            # Calcular densidade
            initial_density = calculate_density(initial_constraints_coefficients)
            
            status = None
            total_workers = 0
            # workers_schedule = 0
            
            # Inicializando uma lista para armazenar os resultados
            statistics_result = []
            iterations_data = []
            initial_relaxation_rate = 0.01
            max_penalty = 10
            
            # Apenas registra a densidade, mas NÃO bloqueia o Solve
            if densidade_aceitavel is not None:
                statistics_result.append(
                    f"Density check: final={final_density:.4f}, threshold={densidade_aceitavel:.4f}"
                )
            
            # Resolver o problema
            status = solver.Solve()
            
            # Ancora a solucao para a fase 2
            best_X = {(d, t): int(round(X[d, t].solution_value())) for d in range(num_periods) for t in range(limit_workers)} \
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else None

            
            
            # ==========================================================
            # PATCH 2 — FASE 2: FIXA MOTORISTAS E BALANCEIA CARGA
            # ==========================================================

            if (radio_selection_object == "Minimize Total Number of Drivers" and status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)):

                best_U = sum(int(U[d].solution_value()) for d in range(num_periods))
                best_Z = sum(int(Z[t].solution_value()) for t in range(limit_workers))

                # 🔒 trava o nível ótimo encontrado na Fase 1
                solver.Add(solver.Sum(U[d] for d in range(num_periods)) == best_U)
                solver.Add(solver.Sum(Z[t] for t in range(limit_workers)) == best_Z)

                if best_X is not None:
                    for (d, t), v in best_X.items():
                        X[d, t].SetBounds(v, v)



                # 🎯 FASE 2 — balanceamento real
                # solver.Minimize(Lmax - L[t] for t in range(limit_workers))
                # solver.Minimize(solver.Sum(Lmax - L[t] for t in range(limit_workers) if t in L))
                
                # ==========================
                # FASE 2 — refino (após fixar U e Z)
                # ==========================
                
                
                SOFT_DRIVING = 32  # 8h = 32 slots
                Excess = {t: solver.NumVar(0, solver.infinity(), f"Excess[{t}]") for t in range(limit_workers)}
                for t in range(limit_workers): solver.Add(Excess[t] >= L[t] - SOFT_DRIVING)
                

                # peso muito baixo só para desempatar (não pode competir com Z, pq Z já está fixo)
                W_Y    = 1.0      # penaliza presença (turno)
                W_OVER = 5.0      # penaliza excesso acima do soft cap (se você quer)
                W_LMAX = 0.1      # suaviza picos de carga por motorista

                total_presence = solver.Sum(Y[d, t] for d in range(num_periods) for t in range(limit_workers))
                # total_over     = solver.Sum(Over[t] for t in range(limit_workers))
                total_over = solver.Sum(Excess[t] for t in range(limit_workers))

                # opcional: reduzir maior carga (mais estável)
                objective_phase2 = (
                    W_Y    * total_presence
                + W_OVER * total_over
                + W_LMAX * Lmax
                )

                solver.Minimize(objective_phase2)
                # resolve novamente
                status = solver.Solve()
            
            # ==========================================================
            # EXTRAÇÃO CANÔNICA DA SOLUÇÃO (🔥 ESSENCIAL 🔥)
            # ==========================================================
            matrix_allocation = None
            workers_schedule = None
            total_workers = 0

            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):

                matrix_allocation = np.array(
                    [
                        # [int(Y[d, t].solution_value()) for t in range(limit_workers)]
                        [1 if Y[d, t].solution_value() > 0.5 else 0 for t in range(limit_workers)]
                        for d in range(num_periods)
                    ],
                    dtype=int
                )

                # trabalhadores por período (slot)
                # workers_schedule = matrix_allocation.sum(axis=1).astype(int).tolist()
                workers_schedule = [sum(Y[d,t].solution_value() for t in range(limit_workers)) for d in range(num_periods)]
                tasks_schedule   = [sum(X[d,t].solution_value() for t in range(limit_workers)) for d in range(num_periods)]                                
                        
                        
                                
                # -----------------------------------------
                # Horas por motorista (para seus gráficos)
                # slot = 15min => 0.25h
                # -----------------------------------------
                # slot_hours = 0.25
                # turno_por_motorista = (matrix_allocation.sum(axis=0) * slot_hours).astype(float).tolist()  
                
                slot_hours = 0.25

                # 1) Turno/presença (Y)
                shift_hours_per_driver = [
                    sum(Y[d, t].solution_value() for d in range(num_periods)) * slot_hours
                    for t in range(limit_workers)
                ]

                # 2) Condução/atendimento (X) em "slots de tarefa" (cap=1 => 1 tarefa = 15min)
                driving_hours_per_driver = [
                    sum(X[d, t].solution_value() for d in range(num_periods)) * slot_hours
                    for t in range(limit_workers)
                ]

                # filtra ativos (Z=1)
                active_idx = [t for t in range(limit_workers) if Z[t].solution_value() > 0.5]

                shift_hours_active   = [shift_hours_per_driver[t] for t in active_idx]
                driving_hours_active = [driving_hours_per_driver[t] for t in active_idx]
                
                total_active_drivers = int(
                    sum(int(Z[t].solution_value()) for t in range(limit_workers))
                )
                
                total_assigned_slots = int(
                    sum(int(Y[d, t].solution_value()) for d in range(num_periods) for t in range(limit_workers))
                )
            
            else:
                # Sem solução viável: não tente ler solution_value()
                matrix_allocation = None
                workers_schedule = None
                total_active_drivers = 0
                total_assigned_slots = 0                
                
            # Iterar para resolver conflitos
            # max_iterations = limit_iteration
            # relaxation_level = limit_level_relaxation  # Relaxa as restrições progressivamente
            # iteration = 0
            
            
            # if radio_selection_object == "Minimize Total Number of Drivers":
            #     max_iterations = 0

            # while status != pywraplp.Solver.OPTIMAL and iteration < max_iterations:
                
            #     # Relaxar a penalização e as restrições conforme a iteração
            #     relaxation_level = iteration * initial_relaxation_rate  # O nível de relaxamento aumenta com o tempo
                
            #     # Ajustar a função objetivo com a penalização relaxada
            #     objective = relax_objective(solver, Y, num_periods, limit_workers, penalty, max_penalty, relaxation_level, need, tolerance_demands, cap_tasks_per_driver_per_slot)

            #     # Capturar dados da iteração
            #     iteration_data = {
            #         "iteration": iteration,
            #         "relaxation_level": float(relaxation_level) if relaxation_level is not None else None,
            #         "status": get_solver_status_description(status),
            #         "objective_value": solver.Objective().Value() if status in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE} else 0,
            #     }
                
            #     # Relaxar restrições relacionadas
            #     if selected_restrictions["limite_diario"]:
            #         relax_restrictions(solver, constraints, relaxation_level)
            #     if selected_restrictions["limite_semanal"]:
            #         relax_restrictions(solver, constraints, relaxation_level)
            #     if selected_restrictions["repouso_diario_minimo"]:
            #         relax_restrictions(solver, constraints, relaxation_level)
        
            #     # Resolver novamente
            #     # status = solver.Solve()

            #     iterations_data.append(iteration_data)
                
            #     iteration += 1
                
            if status == pywraplp.Solver.OPTIMAL:
                workers_schedule = [sum(int(Y[d, t].solution_value()) for t in range(limit_workers)) for d in range(num_periods)]

                # total_workers = solver.Objective().Value()
                # total_workers = sum(int(Z[t].solution_value()) for t in range(limit_workers))
                
                total_active_drivers = int(
                    sum(int(Z[t].solution_value()) for t in range(limit_workers))
                )

                total_assigned_slots = int(
                    sum(int(Y[d, t].solution_value()) for d in range(num_periods) for t in range(limit_workers))
                )

                statistics_result.append(f"Model State: OPTIMAL")

            elif status == pywraplp.Solver.FEASIBLE:
                statistics_result.append(f"Model State: FEASIBLE")
            elif status == pywraplp.Solver.INFEASIBLE:
                statistics_result.append(f"Model State: INFEASIBLE")
            elif status == pywraplp.Solver.UNBOUNDED:
                statistics_result.append(f"Model State: UNBOUNDED")
            elif status == pywraplp.Solver.ABNORMAL:
                statistics_result.append(f"Model State: ABNORMAL")
            elif status == pywraplp.Solver.MODEL_INVALID:
                statistics_result.append(f"Model State: MODEL_INVALID")
            elif status == pywraplp.Solver.NOT_SOLVED:
                statistics_result.append(f"Model State: NOT_SOLVED")
            else:
                statistics_result.append("Model State: NOT_SOLVED")
                total_active_drivers = 0
                total_assigned_slots = 0
            
            total_capacity = sum(
                X[d,t].solution_value()
                for d in range(num_periods)
                for t in range(limit_workers)
            )

            avg_load_per_driver = (
                total_capacity / max(1, total_active_drivers)
            )
            
            #-----------------------------
            # Calculos Restul
            #----------------------------
            total_demand = sum(need)
            
            total_unmet = sum(
                U[d].solution_value()
                for d in range(num_periods)
            )

            coverage = 100.0 * (total_demand - total_unmet) / max(1.0, total_demand)

            
            statistics_result.append(f"Total demand: {total_demand}")
            statistics_result.append(f"Total unmet: {total_unmet}")
            statistics_result.append(f"Coverage (%): {coverage}")
            statistics_result.append(f"Total active drivers: {total_active_drivers}")
            statistics_result.append(f"Total assigned slots: {total_assigned_slots}")
            statistics_result.append(f"Model Type: {tipo_modelo(solver)}")
            statistics_result.append(f"Total Resolution Time: {solver.wall_time()} ms")
            statistics_result.append(f"Total Number of Iterations: {solver.iterations()}")
            statistics_result.append(f"Number of Restrictions: {solver.NumConstraints()}")
            statistics_result.append(f"Number of Variables: {solver.NumVariables()}")
            statistics_result.append(f"AVG Load per Driver: {avg_load_per_driver}")
            statistics_result.append(f"DEBUG: sum Y = {sum(Y[d,t].solution_value() for d in range(num_periods) for t in range(limit_workers))}")
            statistics_result.append(f"DEBUG: sum X = {sum(X[d,t].solution_value() for d in range(num_periods) for t in range(limit_workers))}")
            statistics_result.append(f"DEBUG: sum U = {sum(U[d].solution_value() for d in range(num_periods))}")

            save_data(constraints_coefficients, 'constraints_coefficients.json')
            
        objective_value = solver.Objective().Value()
        best_bound = solver.Objective().BestBound()

        gap = abs(objective_value - best_bound) / max(1.0, abs(objective_value))

        statistics_result.extend([
            f"Objective value: {objective_value:.2f}",
            f"Best bound: {best_bound:.2f}",
            f"Gap (%): {gap * 100:.2f}",
            f"Solve time (s): {solver.wall_time()/1000:.1f}",
            f"Stopped by time/gap: {gap <= GAP_REL or solver.wall_time() >= TIME_LIMIT_SEC * 1000}"
        ])
            
    # Captura final dos logs
    solver_stdout = stdout_buffer.getvalue() if stdout_buffer else ""
    solver_stderr = stderr_buffer.getvalue() if stderr_buffer else ""

    # Anexa os logs no resultado
    solver_logs = {
        "stdout": solver_stdout,
        "stderr": solver_stderr
    }  
    
    return (solver, status, 
            total_active_drivers,      # ← decisão
            total_assigned_slots,      # ← esforço
            workers_schedule,
            tasks_schedule,
            driving_hours_per_driver, 
            constraints_coefficients, 
            initial_density, 
            final_density, 
            statistics_result, 
            msg, 
            iterations_data, 
            matrix_allocation, 
            solver_logs
        )


# Adicionando a restrição de pausa fracionada (divisão da pausa)
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
    constraints_coefficients = None #load_data(cache_path)

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
st.title("Simulator")

# Inicializa a variável default_need vazia
default_need = []
need_input = None
num_periods = None

initial_constraints_coefficients = []

# Carrega os dados do arquivo, se existirem
initial_constraints_coefficients = load_data('initial_constraints_coefficients.json')

# Criando o formulário
# with st.form("paramModel"):
with st.expander("Parameters", expanded=True):    
    
    st.subheader("Profile")

    profile_key = st.selectbox(
        "Select a reference profile (optional)",
        options=["Custom"] + list(PROFILES.keys()),
        format_func=lambda k: "Custom (manual)" if k == "Custom"
        else f"{k} — {PROFILES[k]['description']}"
    )

    # ===============================
    # PROFILE STATE MANAGEMENT
    # ===============================

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
    
    # Colunas do formulário
    col1, col2, col3, col4 = st.columns([1,1,1,4])
    
    with col1:
        st.write("Global")
        
        total_hours = st.number_input("horizon_hours", min_value=1, key="horizon_hours", value=24)
        period_minutes = st.number_input("Slot", min_value=1, key="slot", value=15)
        tolerance_demands = st.number_input("Tolerance Coverage", min_value=0.01, key="tolerance_coverage")
        penalty = st.number_input("Penalty for unmet", min_value=0.01, key="penalty_unmet")
        cap_tasks_per_driver_per_slot = st.number_input("Cap. Tasks per Driver per Slot", min_value=1, key="cap_tasks_per_driver_per_slot") #cap_tasks_per_driver_per_slot
        limit_workers = st.number_input("Drivers (0, no limit)", min_value=1, key="limit_workers", value=100)

    with col2:
        st.write("Algorithm")

        variable_type = st.selectbox("Variable", ["Integer", "Binary", "Continuous"], key="variable_type")
        solver_param_type = st.selectbox("GLOP-LP | SCIP-MIP", ["SCIP", "GLOP"], key="solver")
        acceptable_percentage = st.number_input("Acceptable Density", min_value=0.01, key="acceptable_density")

    with col3:
        st.write("Iterations|Relaxation")

        limit_iteration = st.number_input("Limit Iterations", min_value=0, key="limit_iteration")
        limit_level_relaxation = st.number_input("Relaxation", min_value=0, key="relaxation")
        
    with col4:
        fixar_valores = st.checkbox("Set Values")

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
                # congelar valores
                st.session_state.need_text = ", ".join(map(str, st.session_state.default_need))
            else:
                # gerar novos valores apenas UMA vez
                st.session_state.default_need = np.random.randint(1, 11, size=num_periods).tolist()
                st.session_state.need_text = ", ".join(map(str, st.session_state.default_need))

            st.session_state.last_fixar_state = fixar_valores

        # 3) widget
        need_input = st.text_area(
            "Slot Demand",
            key="need_text",
            height=210
        )

        need = [x.strip() for x in need_input.split(",") if x.strip()]
        st.write(f"Total Demand {len(need)}")

        # parsing seguro
        needNew = []
        for x in need:
            try:
                needNew.append(float(x))
            except ValueError:
                pass
    # with col4:
    #     fixar_valores = st.checkbox("Set Values")

    #     num_periods = (total_hours * 60) // period_minutes

    #     # 1) inicializa default_need só uma vez (para o modo "fixo")
    #     if "default_need" not in st.session_state:
    #         st.session_state.default_need = np.random.randint(1, 11, size=num_periods).tolist()

    #     # 2) controla o texto do widget explicitamente
    #     if "need_text" not in st.session_state:
    #         st.session_state.need_text = ", ".join(map(str, st.session_state.default_need))

    #     # 3) quando DESMARCA, gere novo aleatório e force update no text_area
    #     if not fixar_valores:
    #         new_need = np.random.randint(1, 11, size=num_periods).tolist()
    #         st.session_state.need_text = ", ".join(map(str, new_need))
    #     else:
    #         # quando MARCA, usa o default_need congelado
    #         st.session_state.need_text = ", ".join(map(str, st.session_state.default_need))

    #     need_input = st.text_area("Slot Demand", key="need_text", height=210)

    #     need = [x.strip() for x in need_input.split(",") if x.strip()]
    #     st.write(f"Total Demand {len(need)}")

    #     # parsing seguro (sem isinstance)
    #     needNew = []
    #     for x in need:
    #         try:
    #             needNew.append(float(x))
    #         except ValueError:
    #             pass


        # fixar_valores = st.checkbox("Set Values", value=True)

        # # Inicialização
        # if "default_need" not in st.session_state:
        #     st.session_state.default_need = gerar_valores_aleatorios(
        #         total_hours, period_minutes
        #     )

        # if "need_text" not in st.session_state:
        #     st.session_state.need_text = ', '.join(map(str, st.session_state.default_need))

        # # Lógica correta
        # if fixar_valores:
        #     # mantém o valor fixo
        #     pass
        # else:
        #     # gera NOVA aleatoriedade e ATUALIZA o input
        #     st.session_state.default_need = np.random.randint(
        #         1, 11, size=(total_hours * 60) // period_minutes
        #     ).tolist()
        #     st.session_state.need_text = ', '.join(map(str, st.session_state.default_need))

        # # Input controlado
        # need_input = st.text_area(
        #     "Slot Demand",
        #     st.session_state.need_text,
        #     height=210,
        #     key="need_text"
        # )

        #Exibir a quantidade de elementos em default_need
        # need = [need.strip() for need in need_input.split(',')] 
        # st.write(f"Total Demand {len(need)}")
        
        # # Limpar e converter a lista de 'need'
        # needNew = [float(x) for x in need if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]
        
        # fixar_valores = st.checkbox("Set Values", value=True)

        # if "default_need" not in st.session_state:
        #     st.session_state.default_need = gerar_valores_aleatorios(
        #         total_hours, period_minutes
        #     )

        # if fixar_valores:
        #     default_need = st.session_state.default_need
        # else:
        #     default_need = np.random.randint( 1, 11, size=(total_hours * 60) // period_minutes).tolist()


        # need_input = st.text_area(f"Slot Demand",', '.join(map(str, default_need)), height=210)
        
        # #Exibir a quantidade de elementos em default_need
        # need = [need.strip() for need in need_input.split(',')] 
        # st.write(f"Total Demand {len(need)}")
        
        # # Limpar e converter a lista de 'need'
        # needNew = [float(x) for x in need if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]

    # Seleção das restrições
    st.subheader("Restrictions")
    with st.expander("Global", expanded=True):
        
        selected_restrictions = {}
        col1, col2, col3 = st.columns([2, 4, 5])

        # Radiobuttons na primeira coluna
        with col1:
            
            st.write("Objective Function")
            radio_selection_object = st.radio(
                "Select the objective", 
                options=["Maximize Demand Response", "Minimize Total Number of Drivers"],
                index=0, 
                key="funcao_Objetivo"
            )
            
            st.write("Break Options")
            radio_selection = st.radio(
                "Select pause", 
                options=["None", "45 minutes", "15+30 split", "30+15 split"],
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

    st.subheader("Cache")
    with st.expander("Parameters", expanded=True):
        atualizarFicheiro = st.checkbox("Update File", value=False, help="If enabled, the options below will be disabled.")
        if atualizarFicheiro:
            constraints_coefficients = initial_constraints_coefficients
            save_data(initial_constraints_coefficients, 'initial_constraints_coefficients.json')
            save_data(constraints_coefficients, 'constraints_coefficients.json')



    inputs = prepare_optimization_inputs(
        need_input=need_input,
        total_hours=total_hours,
        period_minutes=period_minutes,
        restrictions=restrictions,
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


    st.subheader("Machine Learning Mode")

    with st.expander("Machine Learning – Training", expanded=False):

        st.markdown("### Training models LightGBM (f₁ e f₂)")
        st.caption("""
        - f₁ → Driver selection model – period\n
        - f₂ → Neighborhood assessment model for LNS\n
        """)
        
        train_clicked = st.button("Train ML models (f₁ & f₂)", key="train_ml")
        if train_clicked:
            st.session_state["run_train_ml"] = True

    if st.session_state.run_train_ml:
        st.info("Starting model training... this may take a few minutes.")
        progress = st.progress(0)
        log = st.empty()

        progress.progress(0.3)
        log.write("Training f₁ and f₂ models...")

        result = run_ml_training()

        progress.progress(1.0)
        log.write("Training completed!")

        st.session_state.train_ml_result = result
        st.session_state.run_train_ml = False    
        
    if st.session_state.train_ml_result:

        if st.session_state.train_ml_result["success"]:
            st.success("Training process completed!")
            render_ml_training_results(
                st.session_state.train_ml_result["results"]
            )
        else:
            st.error("Error during training:")
            st.code(st.session_state.train_ml_result["error"])
                
    with st.expander("Machine Learning – Dataset Builder", expanded=False):
        
        st.markdown("### Generate training datasets for f₁ (assignment) e f₂ (neighborhood)")

        col_ml_1, col_ml_2, col_ml_3 = st.columns(3)

        with col_ml_1:
            num_instances = st.number_input(
                "Number of instances",
                min_value=1,
                max_value=500,
                value=10,
                step=1,
            )
            num_periods_ml = st.number_input(
                "Periods per instance",
                min_value=4,
                max_value=96,
                value=24,
                step=4,
            )

        with col_ml_2:
            limit_workers_ml = st.number_input(
                "Drivers (limit)",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
            )
            max_demands_per_driver_ml = st.number_input(
                "Maximum number of periods per driver",
                min_value=1,
                max_value=500,
                value=20,
                step=1,
            )

        with col_ml_3:
            demand_min = st.number_input(
                "Minimum demand per period",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )
            demand_max = st.number_input(
                "Maximum demand per period",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
            )
            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
            )

        st.caption(
            "Note: Small instances are recommended for resolving MILP in exact mode."
            "(For example, 24 periods and 10 drivers.)."
        )

        dataset_clicked = st.button("Gerar Dataset ML (f₁ & f₂)", key="dataset_ml")
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
            status_text.text("Concluído!")

            st.success(
                "Datasets gerados com sucesso!\n\n"
                f"- f₁: {len(f1_rows_all)} linhas → {os.path.abspath(f1_path)}\n"
                f"- f₂: {len(f2_rows_all)} linhas → {os.path.abspath(f2_path)}"
            )                

            st.info(f"Pasta dos datasets:\n{os.path.abspath(os.path.dirname(f1_path))}")

            if f1_rows_all:
                st.write("Pré-visualização f₁:")
                st.dataframe(
                    pd.DataFrame(f1_rows_all).head()
                )

            if f2_rows_all:
                st.write("Pré-visualização f₂:")
                st.dataframe(
                    pd.DataFrame(f2_rows_all).head()
                )

        st.session_state.run_dataset_ml = False    

    st.subheader("Optimization Mode")
   
    optimization_mode = st.radio(
        "Mode Select:",
        ["Exact", "Heuristic", "LNS"],
        index=0,
        horizontal=True  # opcional, melhora UX
    )    
    
    st.subheader("Elementary Operations")

    with st.expander("Parameters", expanded=True):
        selected_operations = {}   
        for op in elementalOperations:
            selected_operations[op['Key']] = st.checkbox(
                f"{op['Description']} | {op['Formula']}",
                key=f"op_{op['Key']}",
                help="Enable this elementary operation"
            )

        # =========================
        # UI CONDICIONAL
        # =========================

        if st.session_state.get("op_troca_equacoes"):
            st.markdown("### 🔁 Troca de Equações")
            swap_row_1 = st.number_input("Linha 1", min_value=0, key="swap_row_1")
            swap_row_2 = st.number_input("Linha 2", min_value=0, key="swap_row_2")

        if st.session_state.get("op_multiplicacao_por_constante"):
            st.markdown("### ✖ Multiplicação por Constante")
            mult_row = st.number_input("Linha", min_value=0, key="mult_row")
            mult_const = st.number_input("Constante", value=1, key="mult_const")

        if st.session_state.get("op_soma_multiplo_equacao"):
            st.markdown("### ➕ Soma de múltiplo")
            sum_row_base = st.number_input("Linha base", min_value=0, key="sum_row_base")
            sum_row_target = st.number_input("Linha alvo", min_value=0, key="sum_row_target")
            sum_multiplier = st.number_input("Multiplicador", value=1, key="sum_multiplier")
            
        if st.session_state.get("op_soma_multiplo_equacao_automatica"):
            st.markdown("### ➕ Soma de múltiplo automática")
            sum_row_base_auto = st.number_input("Linha base", min_value=0, key="sum_row_base_auto")
            sum_row_target_auto = st.number_input("Linha alvo", min_value=0, key="sum_row_target_auto")
            sum_multiplier_auto = st.number_input("Multiplicador", value=1, key="sum_multiplier_auto")            
    
                        
    # if submit_button:
    if st.button("Run Optimization"):
        
        # Inicializar variáveis para as operações
        swap_rows = None
        multiply_row = None
        add_multiple_rows = None
        iterations_data_result = []
        try:
            # Sempre cria as colunas ANTES de validar need
            col_resultsItIOpI, col_resultsItIOpII, col_resultsItIOpIII = st.columns(3)
            col_resultsI_col = st.columns(1)[0]
            col_resultsItI_col = st.columns(1)[0]

            need = list(map(int, need_input.split(',')))

            # Exibir a matriz de restrições antes da otimização
            if len(need) != num_periods:
                st.error(f"The input must have exactly {num_periods} values ​​(1 for each period of {period_minutes} minutes).")
            else:
                with col_resultsI_col:
                    # st.subheader("Operações Elementares")
                    # with st.expander("Operations", expanded=True):
                        
                    swap_rows_c = selected_operations["troca_equacoes"] #st.checkbox("Troca de Equações")
                    multiply_row_c = selected_operations["multiplicacao_por_constante"] #st.checkbox("Multiplicação por Constante")
                    add_multiple_rows_c = selected_operations["soma_multiplo_equacao"] #st.checkbox("Somar Múltiplo de uma Equação a Outra")
                    add_multiple_rows_c_auto = selected_operations["soma_multiplo_equacao_automatica"] #st.checkbox("Somar Múltiplo de uma Equação a Outra - Automático")
                    
                    if (not swap_rows_c 
                        and not multiply_row_c 
                            and not add_multiple_rows_c
                            and not add_multiple_rows_c_auto):
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
                                needNew,
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
                                radio_selection_object=radio_selection_object                                
                            )                               
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
                                    needNew,
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
                                    radio_selection_object=radio_selection_object                                                               
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
                                        needNew,
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
                                        radio_selection_object=radio_selection_object                                                                
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
                                    needNew,
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
                                    radio_selection_object=radio_selection_object                                                                
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
                                    needNew,
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
                                    radio_selection_object=radio_selection_object                                                                
                                )                            

                                # Dentro da sua função solve_shift_schedule
                                final_density = calculate_density(constraints_coefficients)
                                
                                if final_density <= acceptable_percentage:
                                    st.write(f"Final density {final_density} has reached acceptable limit ({acceptable_percentage}). Exiting loop.")
                                    break

                    # 🔧 Normalização para modos Heuristic/LNS
                    if "workers_schedule" in locals() and "matrix_allocation" in locals():
                        if workers_schedule is None and matrix_allocation is not None:
                            try:
                                workers_schedule = list(np.sum(matrix_allocation, axis=1))
                            except Exception:
                                workers_schedule = []
                                
                    #-----------------------------------
                    # Tabela de Resultados
                    # ----------------------------------            
                    with st.expander("Results", expanded=True):
                        
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
                            return df

                        # Aplicar estilização
                        styled_df = results_df.style.apply(highlight_cell, axis=None)

                        # Exibir o DataFrame como tabela
                        st.table(styled_df)                                
                             
                    #----------------------------------
                    # Interpretador
                    #----------------------------------         
                           
                    with st.expander("Interpretador", expanded=False):                             
                        interp = interpret_solver_result(statistics_result)
                        st.markdown("## 🧠 Interpretação do Resultado")
                        st.success(interp["model_state"])

                        st.markdown("### 📊 Eficiência Operacional")
                        st.json(interp["operational_metrics"])

                        st.markdown("### ⚙️ Estrutura do Modelo")
                        st.info(interp["model_structure"])

                        st.markdown("### 🧮 Qualidade da Solução")
                        for q in interp["solution_quality"]:
                            st.write("•", q)

                        st.markdown("### ⏱️ Comportamento do Solver")
                        st.warning(interp["solver_behavior"])
                        
                        st.subheader("📐 Estrutura do Modelo")

                        for line in interp["model_structure"]:
                            st.info(line)
                        
                        st.subheader("⚙️ Comportamento do Solver")

                        for line in interp.get("solver_behavior", []):
                            st.warning(line)
                             
                    st.divider()  
                         

                    #--------------------------
                    # Esforço Operacional
                    #--------------------------                         
                    with st.expander("Esforço Operacional", expanded=True):     
                        col_A,col_B, col_C = st.columns(3)                                        
                        # ==========================================================
                        # GRÁFICO 1 — ESFORÇO OPERACIONAL (OFICIAL)
                        # ==========================================================
                        with col_A:
                            
                            slots_por_motorista = matrix_allocation.sum(axis=0)
                            slots_motoristas_ativos = slots_por_motorista[slots_por_motorista > 0]

                            slot_minutes = 15  # ou o valor do seu modelo
                            horas_por_motorista = slots_motoristas_ativos * (slot_minutes / 60.0)

                            
                            fig, resumo = plot_painel_esforco_operacional(horas_por_motorista)
                            st.pyplot(fig)

                            st.caption(
                                f"""
                                Motoristas ativos: **{resumo['motoristas_ativos']}**  
                                Esforço médio: **{resumo['media_horas']:.2f} h**  
                                Mediana: **{resumo['mediana_horas']:.2f} h**  
                                P90 / P95: **{resumo['percentil_90']:.2f} h / {resumo['percentil_95']:.2f} h**  
                                Índice de Gini: **{resumo['indice_gini']:.3f}**

                                ⚠️ O painel apresenta indicadores de **esforço operacional agregado**.
                                A referência de 9h é indicativa e não representa, isoladamente,
                                verificação direta de conformidade legal.
                                """
                            )

                        
                        # with col_B:
                        #     # st.subheader("📦 Esforço Operacional")

                        #     plot_operational_effort_per_driver(
                        #         matrix_allocation=matrix_allocation,
                        #         slot_minutes=15,
                        #         daily_limit_hours=9.0
                        #     )

                        with col_B:
                            # st.subheader("📦 Esforço Operacional Motoristas Ativos")
                            
                            plot_operational_effort_per_driver_active(
                                matrix_allocation=matrix_allocation
                            )

                        with col_C:
                            
                            # se SHIFT_MAX no modelo está em slots (ex.: 52)
                            SHIFT_MAX_HOURS = 13   # 13h

                            
                            # motoristas ativos (têm pelo menos 1 slot em Y)
                            active_drivers_idx = [
                                t for t in range(matrix_allocation.shape[1])
                                if matrix_allocation[:, t].sum() > 0
                            ]

                            # ---- Painel A: Condução (X) ----
                            driving_hours_active = [
                                driving_hours_per_driver[t]
                                for t in active_drivers_idx
                            ]

                            # ---- Painel B: Turno (Y) ----
                            slot_hours = 0.25
                            shift_hours_active = [
                                matrix_allocation[:, t].sum() * slot_hours
                                for t in active_drivers_idx
                            ]
                                                        
                            
                            
                            st.subheader("📘 Painel A — Condução (UE)")

                            fig_a = plot_panel_a_driving(driving_hours_active)
                            st.pyplot(fig_a)

                            st.divider()

                            st.subheader("📗 Painel B — Turno Operacional")

                            fig_b = plot_panel_b_shift(
                                shift_hours_active,
                                SHIFT_MAX=SHIFT_MAX_HOURS
                            )
                            st.pyplot(fig_b)



                    #--------------------------
                    # Esforço Operacional
                    #--------------------------  
                    if msg is not None:
                        # Converter a entrada de texto para uma lista de números
                        try:
                            demanda = list(map(int, need_input.split(',')))
                        except ValueError:
                            st.error("Por favor, insira os valores da demanda separados por vírgula e espaço.")
                            demanda = []                       
                    
                                           
                    with st.expander("Gráficos Demanda and Gaps", expanded=True):     
                        col_A,col_B = st.columns(2) 

                        with col_A:
                            # st.subheader("Results")

                            # with st.expander("Demanda vs Motoristas Alocados", expanded=True):
                                # st.pyplot(plot_demand_vs_drivers_line(demanda, workers_schedule))
                                
                                
                            # drivers_present_per_slot = [
                            #     sum(1 for t in range(limit_workers) if Y[d, t].solution_value() > 0.5)
                            #     for d in range(num_periods)
                            # ]

                            # tasks_served_per_slot = [
                            #     sum(X[d, t].solution_value() for t in range(limit_workers))
                            #     for d in range(num_periods)
                            # ]

                            fig = plot_demand_capacity_utilization(
                                demanda=need,
                                workers_schedule=workers_schedule,
                                tasks_schedule=tasks_schedule,          # <<< X agregado
                                cap_tasks_per_driver_per_slot=cap_tasks_per_driver_per_slot
                            )

                                
                                
                                
                                # fig = plot_demand_capacity_utilization(
                                #     demanda=need,
                                #     matrix_allocation=matrix_allocation,
                                #     cap_tasks_per_driver_per_slot=cap_tasks_per_driver_per_slot
                                # )
                                # st.pyplot(fig)


                            # fig2 = plot_demand_vs_capacity(demanda=need,
                            #                                     matrix_allocation=matrix_allocation,
                            #                                     cap_tasks_per_driver_per_slot=cap_tasks_per_driver_per_slot
                            #                                 )
                            # st.pyplot(fig2)
                            
                            
                            fig2 = plot_demand_vs_capacity(
                                demanda=need,
                                workers_schedule=workers_schedule,          # ← vem do solver (Y)
                                tasks_schedule=tasks_schedule,              # ← vem do solver (X)
                                cap_tasks_per_driver_per_slot=cap_tasks_per_driver_per_slot
                            )
                            st.pyplot(fig2)


                            fig3 = plot_capacity_utilization(
                                                                demanda=need,
                                                                matrix_allocation=matrix_allocation,
                                                                cap_tasks_per_driver_per_slot=cap_tasks_per_driver_per_slot
                                                            )
                            st.pyplot(fig3)


                        
                        with col_B:
                            
                            # with st.expander("Gap de Cobertura por Período", expanded=True):
                                fig_gap = plot_gap_chart(demanda, workers_schedule)
                                st.pyplot(fig_gap)

                    # Exibir resultados na primeira coluna
                    with col_resultsItI_col:
                            if msg is not None:
                                    col_A_, col_B_ = st.columns(2)
                                    
                                    # Gerar DataFrame para os dados
                                    slots = list(range(1, len(demanda) + 1))  # Slots de 1 a 96
                                    df_comparacao = pd.DataFrame({
                                        "Slot": slots,
                                        "Demanda": demanda,
                                        "Motoristas": workers_schedule
                                    })
                                    
                                    df = df_comparacao.copy()

                                    # Evitar divisões inválidas
                                    df["Demanda_Pos"] = df["Demanda"].replace(0, np.nan)

                                    # Métricas fundamentais
                                    df["Coverage Ratio"] = df["Motoristas"] / df["Demanda_Pos"]
                                    df["Gap"] = df["Motoristas"] - df["Demanda"]

                                    # Severidade (sempre >= 0)
                                    df["Deficit"] = df["Gap"].apply(lambda x: -x if x < 0 else 0)
                                    df["Excess"] = df["Gap"].apply(lambda x: x if x > 0 else 0)

                                    # Flag binária
                                    df["Violation"] = df["Gap"] < 0
                                    
                                    slots_totais = len(df)
                                    slots_violados = int(df["Violation"].sum())

                                    # Estabilidade temporal
                                    coverage_std = float(np.nanstd(df["Coverage Ratio"]))

                                    # Severidade
                                    deficit_total = float(df["Deficit"].sum())
                                    deficit_max = float(df["Deficit"].max())

                                    # Cobertura ponderada pela demanda (métrica científica)
                                    weighted_coverage = (
                                        np.minimum(df["Motoristas"], df["Demanda"]).sum()
                                        / df["Demanda"].sum()
                                    )

                                    with col_A_:
                                        col_A,col_B,col_C = st.columns(3)
                                        with col_A:
                                            with st.expander("Stability Analysis", expanded=True):
                                                st.metric(
                                                    "Coverage Stability (σ)",
                                                    f"{coverage_std:.2f}",
                                                    help="Desvio padrão da razão Motoristas/Demanda ao longo dos slots."
                                                )

                                                # if coverage_std < 0.2:
                                                #     st.success("Stable allocation")
                                                # elif coverage_std < 0.5:
                                                #     st.warning("Moderately unstable allocation")
                                                # else:
                                                #     st.error("Highly unstable allocation")
                                                if radio_selection_object == "Minimize Total Number of Drivers":

                                                    # Neste regime, alta variância é esperada e não é erro
                                                    if coverage_std < 0.5:
                                                        st.info("Capacity-driven allocation (expected variability)")
                                                    else:
                                                        st.warning(
                                                            "High capacity dispersion across periods — expected under driver-minimization regime"
                                                        )

                                                elif radio_selection_object == "Maximize Demand Response":

                                                    # Aqui sim usamos estabilidade clássica
                                                    if coverage_std < 0.2:
                                                        st.success("Stable allocation")
                                                    elif coverage_std < 0.5:
                                                        st.warning("Moderately unstable allocation")
                                                    else:
                                                        st.error("Highly unstable allocation")

                                                else:
                                                    # Regime híbrido ou experimental
                                                    if coverage_std < 0.2:
                                                        st.success("Stable allocation")
                                                    elif coverage_std < 0.5:
                                                        st.warning("Moderate variability detected")
                                                    else:
                                                        st.warning("High variability — review trade-offs")
                                                

                                                # Exibir o desvio padrão da taxa de cobertura com cor e formatação em negrito
                                                # st.markdown(f"<h3 style='color:{desvio_cor}; font-weight: bold; font-size: 16px;'>Standard Deviation Coverage Rate: {desvio_padrao:.2f} ({status})</h3>", unsafe_allow_html=True)

                                                # Calcular e exibir os totais para cada indicador
                                                total_demanda = df_comparacao['Demanda'].sum()
                                                total_motoristas = df_comparacao['Motoristas'].sum()
                                                
                                                total_excess = float(df["Excess"].sum())

                                                # Estabilidade
                                                coverage_std = float(np.nanstd(df["Coverage Ratio"]))

                                                # Frequência
                                                slots_violados = int(df["Violation"].sum())
                                                slots_totais = len(df)

                                                # Severidade
                                                weighted_coverage = (
                                                    np.minimum(df["Motoristas"], df["Demanda"]).sum()
                                                    / df["Demanda"].sum()
                                                )

                                                # total_sobrecarga = df_comparacao['Índice de Sobrecarga'].sum()
                                                total_excess = float(df["Excess"].sum())

                                                # total_subutilizacao = df_comparacao['Índice de Subutilização'].sum()
                                        with col_B:
                                            
                                            kpis = compute_coverage_kpis(demanda, tasks_schedule)

                                            kpis["slots_with_deficit"]
                                            kpis["total_uncovered"]
                                            kpis["max_deficit"]
                                            kpis["weighted_coverage_pct"]

                                            
                                            st.metric(
                                                "Slots with Demand Deficit",
                                                slots_violados,
                                                help="Número de slots em que a demanda não foi totalmente atendida."
                                            )

                                            st.metric(
                                                "Fully Covered Slots (%)",
                                                round(100 * (slots_totais - slots_violados) / slots_totais, 2),
                                                help="Percentual de slots com cobertura total."
                                            )

                                            st.metric(
                                                "Weighted Coverage Rate (%)",
                                                round(100 * weighted_coverage, 2),
                                                help="Cobertura ponderada pela demanda total."
                                            )
      
                                        with col_C:
                                            st.metric(
                                                "Total Uncovered Demand",
                                                int(deficit_total),
                                                help="Déficit total acumulado ao longo do horizonte."
                                            )

                                            st.metric(
                                                "Max Deficit (single slot)",
                                                int(deficit_max),
                                                help="Maior déficit observado em um único slot."
                                            )
                                            
                                            st.metric(
                                                    "Total Excess Capacity",
                                                    int(total_excess),
                                                    help="Capacidade excedente acumulada ao longo dos slots."
                                            )
                                     
                                    with col_B_:
                                        # fig1 = plot_demand_vs_capacity_with_violations(df)
                                        # st.pyplot(fig1)

                                        fig2 = plot_comparative_capacity_analysis(df)
                                        st.pyplot(fig2)

                                        # with st.expander(f"Demand vs. Drivers (Total Demand: {total_demanda}, Total Drivers: {total_motoristas})", expanded=True):
                                        #     # Exibir gráfico com barras comparativas
                                        #     st.bar_chart(df_comparacao
                                        #                 .set_index("Slot")[["Demanda", "Motoristas"]]
                                        #                 # .reset_index()
                                        #                 .rename(columns={
                                        #                     "Demanda": "Total Demands",
                                        #                     "Motoristas": "Assigned Drivers"
                                        #                 })
                                        #             )

                                    # with col_C:
                                    #     st.metric(
                                    #         "Slots with Demand Deficit",
                                    #         slots_violados,
                                    #         help="Número de slots em que a demanda não foi totalmente atendida."
                                    #     )

                                    #     st.metric(
                                    #         "Fully Covered Slots (%)",
                                    #         round(100 * (slots_totais - slots_violados) / slots_totais, 2),
                                    #         help="Percentual de slots com cobertura total."
                                    #     )

                                    #     st.metric(
                                    #         "Weighted Coverage Rate (%)",
                                    #         round(100 * weighted_coverage, 2),
                                    #         help="Cobertura ponderada pela demanda total."
                                    #     )

                                    #     st.metric(
                                    #         "Total Uncovered Demand",
                                    #         int(deficit_total),
                                    #         help="Déficit total acumulado ao longo do horizonte."
                                    #     )

                                    #     st.metric(
                                    #         "Max Deficit (single slot)",
                                    #         int(deficit_max),
                                    #         help="Maior déficit observado em um único slot."
                                    #     )
                                        
                                    #     st.metric(
                                    #             "Total Excess Capacity",
                                    #             int(total_excess),
                                    #             help="Capacidade excedente acumulada ao longo dos slots."
                                    #     )


                                        # df_gap = (
                                        #     df_comparacao
                                        #     .assign(Gap=lambda d: d["Motoristas"] - d["Demanda"])
                                        #     .set_index("Slot")[["Gap"]]
                                        # )

                                        # violations = (df_gap["Gap"] < 0).sum()

                                        # st.metric(
                                        #     label="Slots with Demand Violation",
                                        #     value=int(violations)
                                        # )

                                        # st.metric(
                                        #     label="Coverage Rate (%)",
                                        #     value=round(100 * (df_gap["Gap"] >= 0).mean(), 2)
                                        # )
                                        
                                    # ============================================================
                                    # NOVOS GRÁFICOS COMPLEMENTARES (gap/risco) — Fase 5 (UI)
                                    # ============================================================
                                    try:
                                        # Garantir tipos numéricos
                                        df_comparacao["Demanda"] = pd.to_numeric(df_comparacao["Demanda"], errors="coerce").fillna(0)
                                        df_comparacao["Motoristas"] = pd.to_numeric(df_comparacao["Motoristas"], errors="coerce").fillna(0)

                                        # Métricas novas por slot
                                        df_comparacao["Safety Margin"] = df_comparacao["Motoristas"] - df_comparacao["Demanda"]
                                        df_comparacao["Unmet Demand"] = (df_comparacao["Demanda"] - df_comparacao["Motoristas"]).clip(lower=0)
                                        df_comparacao["Overstaffing"] = (df_comparacao["Motoristas"] - df_comparacao["Demanda"]).clip(lower=0)

                                        total_unmet = float(df_comparacao["Unmet Demand"].sum())
                                        total_over = float(df_comparacao["Overstaffing"].sum())
                                        min_margin = float(df_comparacao["Safety Margin"].min()) if len(df_comparacao) else 0.0

                                        # with st.expander(f"Safety Margin por Slot (Min: {min_margin:.2f})", expanded=True):
                                        #     st.line_chart(df_comparacao.set_index("Slot")["Safety Margin"])

                                    except Exception as _e:
                                        st.info(f"Gráficos adicionais (gap/risco) indisponíveis: {_e}")
                                            
                                            
                                    # # ============================================================
                                    # # NOVOS GRÁFICOS (distribuições) — coverage e carga por motorista
                                    # # ============================================================
                                    # try:
                                    #     col_dist_1, col_dist_2 = st.columns(2)

                                    #     # 1) Distribuição da taxa de cobertura (Taxa de Cobertura já existe no df_comparacao)
                                    #     with col_dist_1:
                                    #         with st.expander("Distribuição da Taxa de Cobertura (histograma)", expanded=True):
                                    #             fig, ax = plt.subplots()
                                    #             series_cov = pd.to_numeric(df_comparacao.get("Taxa de Cobertura", pd.Series([], dtype=float)), errors="coerce").dropna()
                                    #             if len(series_cov) == 0:
                                    #                 ax.text(0.5, 0.5, "Sem dados de cobertura.", ha="center", va="center")
                                    #             else:
                                    #                 ax.hist(series_cov.values, bins=min(30, max(5, int(len(series_cov) / 2))))
                                    #                 ax.set_xlabel("Taxa de Cobertura (Motoristas/Demanda)")
                                    #                 ax.set_ylabel("Frequência")
                                    #                 ax.set_title("Distribuição da Taxa de Cobertura")
                                    #             # st.pyplot(fig)


                                    #     # 2) Carga por motorista (precisa do matrix_allocation)
                                    #     with col_dist_2:
                                    #         with st.expander("Carga por Motorista (histograma)", expanded=True):
                                    #             if matrix_allocation is None:
                                    #                 st.info("Indisponível: matrix_allocation é None.")
                                    #             else:
                                    #                 try:
                                    #                     mat = np.asarray(matrix_allocation, dtype=int)
                                    #                     # soma por motorista (colunas = drivers)
                                    #                     load_per_driver = mat.sum(axis=0)

                                    #                     fig, ax = plt.subplots()
                                    #                     ax.hist(load_per_driver, bins=min(30, max(5, int(len(load_per_driver) / 2))))
                                    #                     ax.set_xlabel("Total de slots atribuídos ao motorista")
                                    #                     ax.set_ylabel("Nº de motoristas")
                                    #                     ax.set_title("Distribuição de carga por motorista")
                                    #                     # st.pyplot(fig)

                                    #                     st.caption(
                                    #                         f"Min={int(load_per_driver.min())} | "
                                    #                         f"Mediana={float(np.median(load_per_driver)):.1f} | "
                                    #                         f"Max={int(load_per_driver.max())}"
                                    #                     )
                                    #                 except Exception as _e2:
                                    #                     st.info(f"Histograma de carga por motorista indisponível: {_e2}")

                                    # except Exception as _e:
                                    #     st.info(f"Gráficos de distribuições indisponíveis: {_e}")

                                    col_dist_1_, col_dist_2_, col_dist_3_ = st.columns(3)
                                    with col_dist_1_:
                                        # Carga por motorista em slots
                                        slots_por_motorista = matrix_allocation.sum(axis=0)

                                        # Separação conceitual
                                        ativos = slots_por_motorista[slots_por_motorista > 0]
                                        inativos = slots_por_motorista[slots_por_motorista == 0]

                                        fig = plot_carga_motoristas_ativos(ativos)
                                        st.pyplot(fig)




                                    # ============================================================
                                    # NOVO HEATMAP (drivers × slots) — visual da solução
                                    # ============================================================
                                    with col_dist_2_:
                                        st.metric("Motoristas Ativos", len(ativos))
                                        st.metric("Motoristas Inativos", len(inativos))
                                        st.metric("Carga média (ativos)", round(ativos.mean(), 2))
                                        st.metric("Carga máxima", int(ativos.max()))

                                    with col_dist_3_:    
                                        pass
                                    
                                    with st.expander("Heatmap da Solução (Drivers × Slots)", expanded=True):
                                        if matrix_allocation is None:
                                            st.info("Indisponível: matrix_allocation é None.")
                                        else:
                                            try:
                                                mat = np.asarray(matrix_allocation, dtype=int)

                                                # # Segurança: se estiver muito grande, limitar visualização
                                                # max_periods_view = 600
                                                # max_drivers_view = 600

                                                # view = mat[:max_periods_view, :max_drivers_view]

                                                # fig, ax = plt.subplots(figsize=(12, 4))
                                                # sns.heatmap(view.T, ax=ax, cbar=True)  # transpose p/ (drivers no eixo Y)
                                                # ax.set_xlabel("Períodos (slots)")
                                                # ax.set_ylabel("Motoristas")
                                                # ax.set_title("Alocação (1=atribuído) — recorte para visualização")

                                                # st.pyplot(fig)

                                                # if mat.shape[0] > max_periods_view or mat.shape[1] > max_drivers_view:
                                                #     st.caption(
                                                #         f"Mostrando recorte: períodos 0..{max_periods_view-1}, "
                                                #         f"motoristas 0..{max_drivers_view-1}. "
                                                #         f"Shape total={mat.shape}."
                                                #     )
                                                    
                                                    
                                                from matplotlib.colors import ListedColormap

                                                cmap = ListedColormap(["white", "#1f78b4"])

                                                fig, ax = plt.subplots(figsize=(14, 6))

                                                sns.heatmap(
                                                    mat.T,
                                                    ax=ax,
                                                    cmap=cmap,
                                                    cbar=True
                                                    # ,                                                         linewidths=0.0
                                                )

                                                ax.set_xlabel("Períodos (slots de 15 min)")
                                                ax.set_ylabel("Motoristas")
                                                ax.set_title("Mapa de Alocação Binária (recorte)")

                                                st.pyplot(fig)
                                                    
                                                
                                                # slots_per_day = 96
                                                # T, D = mat.shape
                                                # T2 = (T // slots_per_day) * slots_per_day
                                                # mat_trim = mat[:T2, :]

                                                # mat_days = mat_trim.reshape(
                                                #     T2 // slots_per_day,
                                                #     slots_per_day,
                                                #     D
                                                # ).sum(axis=1) * 0.25  # horas

                                                # fig2, ax = plt.subplots(figsize=(12, 6))
                                                # sns.heatmap(
                                                #     mat_days.T,
                                                #     cmap="YlGnBu",
                                                #     ax=ax
                                                # )

                                                # ax.set_xlabel("Dias")
                                                # ax.set_ylabel("Motoristas")
                                                # ax.set_title("Horas Trabalhadas por Motorista e Dia")

                                                # st.pyplot(fig2)
                                                    

                                                # slots_per_day = 96
                                                # T, D = mat.shape
                                                # T2 = (T // slots_per_day) * slots_per_day
                                                # mat_trim = mat[:T2, :]

                                                # mat_days = mat_trim.reshape(
                                                #     T2 // slots_per_day,
                                                #     slots_per_day,
                                                #     D
                                                # ).sum(axis=1) * 0.25  # horas

                                                # fig, ax = plt.subplots(figsize=(12, 6))
                                                # sns.heatmap(
                                                #     mat_days.T,
                                                #     cmap="YlGnBu",
                                                #     ax=ax
                                                # )

                                                # ax.set_xlabel("Dias")
                                                # ax.set_ylabel("Motoristas")
                                                # ax.set_title("Horas Trabalhadas por Motorista e Dia")

                                                # st.pyplot(fig)
                                                    
                                                    

                                            except Exception as _e:
                                                st.info(f"Heatmap 2D indisponível: {_e}")
                                                        
                col_resultsIniI, col_resultsIniII = st.columns(2)
                with col_resultsIniI:
                    if msg is not None:    
                        if initial_density_matrix is not None:
                            
                            # Exibir a densidade
                            st.write(f"Density: {initial_density_matrix:.4f}")
                            with st.expander("Initial Constraint Matrix", expanded=True):
                                fig, ax = plt.subplots(figsize=(14, 8))
                                sns.heatmap(initial_constraints_coefficients, cmap="Blues", cbar=False, annot=False, fmt="d", annot_kws={"size": 7})
                                plt.title('Constraint Matrix')
                                plt.xlabel('X')
                                plt.ylabel('Period')
                                st.pyplot(fig)
                        
                        # Converter dados para DataFrame
                    if iterations_data_result != []:
                        st.subheader("Convergence Progress")
                        df_iterationsResult = pd.DataFrame(iterations_data_result)
                        
                        # 🔒 Garantir colunas mínimas esperadas
                        if "objective_value" not in df_iterationsResult.columns:
                            df_iterationsResult["objective_value"] = np.nan

                        if "iteration" not in df_iterationsResult.columns:
                            df_iterationsResult["iteration"] = range(len(df_iterationsResult))
                        
                        if "relaxation_level" not in df_iterationsResult.columns:
                            df_iterationsResult["relaxation_level"] = range(len(df_iterationsResult))

                        # # Gráfico de convergência do objetivo
                        fig, ax = plt.subplots(figsize=(14, 8))
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
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel("Goal Value")
                        ax.grid(True)
                        ax.legend()
                        
                        # Exibir o gráfico no Streamlit
                        st.pyplot(fig)    

                        if "improved" in df_iterationsResult.columns:
                            fig, ax = plt.subplots(figsize=(14, 4))

                            df_iterationsResult["improved_int"] = df_iterationsResult["improved"].astype(int)

                            ax.bar(
                                df_iterationsResult["iteration"],
                                df_iterationsResult["improved_int"],
                                color=["green" if x == 1 else "lightgray" for x in df_iterationsResult["improved_int"]]
                            )
                            st.subheader("LNS – Improvement")
                            ax.set_title("LNS – Improvement per Iteration")
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel("Improved (1 = Yes)")
                            ax.set_yticks([0, 1])
                            ax.grid(True)

                            st.pyplot(fig)
                            
                        if "total_workers" in df_iterationsResult.columns:
                            fig, ax = plt.subplots(figsize=(14, 5))

                            df_iterationsResult.plot(
                                x="iteration",
                                y="total_workers",
                                ax=ax,
                                marker="s",
                                color="purple",
                                label="Total Workers"
                            )

                            st.subheader("LNS – Workers Evolution")
                            ax.set_title("LNS – Total Workers Evolution")
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel("Total Workers")
                            ax.grid(True)
                            ax.legend()

                            st.pyplot(fig)
                            
                with col_resultsIniII:
                    if msg is not None:
                        if final_density is not None:
                            st.write(f"Final Density Matrix Constraints: {final_density:.4f}")
                        else:
                            st.write("Final Density Matrix Constraints: not computed for this mode.")
                        with st.expander("Final Constraints Matrix", expanded=True):
                            figNew, axNew = plt.subplots(figsize=(14, 8))
                            constraints_coefficients = load_data('constraints_coefficients.json')
                            sns.heatmap(constraints_coefficients, cmap="Oranges", cbar=False, annot=False, fmt="d", annot_kws={"size": 6})
                            plt.title('Constraints Matrix')
                            plt.xlabel('X')
                            plt.ylabel('Period')
                            st.pyplot(figNew)
                            
                    if iterations_data_result != []:
                                    st.subheader("Relaxation Progress")
                                    fig_relax, ax_relax = plt.subplots(figsize=(14, 8))
                                    df_iterationsResult.plot(x="iteration", y="relaxation_level", ax=ax_relax, marker="x", color="red", label="Relaxation Level")
                                    ax_relax.set_title("Relaxation Progress")
                                    ax_relax.set_xlabel("Iteration")
                                    ax_relax.set_ylabel("Relaxation Level")
                                    ax_relax.grid(True)
                                    ax_relax.legend()

                                    # Exibir o gráfico no Streamlit
                                    st.pyplot(fig_relax)                    
                
                if iterations_data_result != []:            
                    st.table(df_iterationsResult)
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
                    
            st.header("📊 Advanced KPIs of the Solution")

            kpis, coverage, safety_margin = compute_kpis(
                demanda,
                workers_schedule=workers_schedule,
                matrix_allocation=matrix_allocation,
            )

            colA, colB, colC = st.columns(3)

            with colA:
                
                global_cov = kpis.get("global_coverage")

                if isinstance(global_cov, (int, float)):
                    st.metric("Global Coverage Score", f"{global_cov:.3f}")
                else:
                    st.metric("Global Coverage Score", "N/A")
                
                we = kpis.get("worker_efficiency")
                st.metric("Worker Efficiency", f"{we:.3f}" if we is not None else "N/A")

            with colB:
                op_risk = kpis.get("operational_risk")
                st.metric("Operational Risk (%)", f"{op_risk*100:.1f}%" if op_risk is not None else "N/A")

                risk_sev = kpis.get("operational_risk_severity")
                st.metric("Risk Severity", int(risk_sev) if risk_sev is not None else "N/A")

            with colC:
                cost = kpis.get("cost_index")
                st.metric("Estimated Cost (€)", f"{cost:.2f}" if cost is not None else "N/A")

                stability = kpis.get("temporal_stability")
                st.metric("Temporal Stability", f"{stability:.3f}" if stability is not None else "N/A")

            st.divider()

            # ---------------------------------------------------------
            # Heatmaps analíticos
            # ---------------------------------------------------------
            st.subheader("Analytical Heatmaps")
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
                    st.pyplot(heat1)
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
                    st.pyplot(heat2)
                else:
                    st.info("Safety margin heatmap indisponível para esta solução.")            
                
            # ---------------------------------------------------------
            # Radar chart (comparing modes)
            # ---------------------------------------------------------
            colA2,colB2 = st.columns(2)

            with colA2:
                def safe_value(x, default=0.0):
                    return x if isinstance(x, (int, float)) else default

                def safe_invert(x, default=0.0):
                    return 1 - x if isinstance(x, (int, float)) else default


                if optimization_mode in ["Exact", "Heuristic", "LNS"]:
                    st.subheader("📈 Comparison between Modes")

                    radar = plot_kpi_radar({
                        "GlobalCoverage": safe_value(kpis.get("global_coverage", 0.0)),
                        "Efficiency": safe_value(kpis.get("worker_efficiency", 0.0)),
                        "Stability": safe_value(kpis.get("temporal_stability", 0.0)),
                        "Risk": safe_invert(kpis.get("operational_risk", 0.0)),
                    })

                    st.pyplot(radar)

            with colB2:
                pass

            st.divider()
                    
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
        


