# # ================================================================
# # Gráficos avançados do dashboard Ottimizia
# # ================================================================

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# def plot_heatmap_safety(safety_margin):
#     fig, ax = plt.subplots(figsize=(16, 2))
#     sns.heatmap([safety_margin], cmap="coolwarm", center=0, cbar=True, ax=ax)
#     ax.set_title("Safety Margin (Motoristas - Demanda)")
#     ax.set_xlabel("Slot")
#     ax.set_ylabel("")
#     return fig

# def plot_heatmap_coverage(coverage):
#     fig, ax = plt.subplots(figsize=(16, 2))
#     sns.heatmap([coverage], cmap="Greens", center=1, cbar=True, ax=ax)
#     ax.set_title("Coverage Ratio por Slot")
#     ax.set_xlabel("Slot")
#     ax.set_ylabel("")
#     return fig

# def plot_kpi_radar(kpis_dict):
#     import numpy as np

#     labels = list(kpis_dict.keys())
#     values = list(kpis_dict.values())

#     values += values[:1]
#     angles = np.linspace(0, 2 * np.pi, len(values))

#     fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
#     ax.plot(angles, values, linewidth=2)
#     ax.fill(angles, values, alpha=0.3)

#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=8)
#     ax.set_title("Comparação dos KPIs por Modo")

#     return fig

# ================================================================
# Gráficos avançados do dashboard (versão segura)
# ================================================================

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt


# ------------------------------------------------
# Heatmap – Safety Margin (por slot)
# ------------------------------------------------
# def plot_heatmap_safety(safety_margin):
#     fig, ax = plt.subplots(figsize=(16, 2))

#     if safety_margin is None or len(safety_margin) == 0:
#         ax.text(0.5, 0.5, "Safety Margin N/A",
#                 ha="center", va="center", fontsize=12)
#         ax.set_axis_off()
#         return fig

#     data = np.array(safety_margin, dtype=float).reshape(1, -1)

#     sns.heatmap(
#         data,
#         cmap="coolwarm",
#         center=0,
#         cbar=True,
#         ax=ax
#     )

#     ax.set_title("Safety Margin (Motoristas − Demanda)")
#     ax.set_xlabel("Slot")
#     ax.set_yticks([])

#     return fig

# def plot_heatmap_safety(safety_margin):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     fig, ax = plt.subplots(figsize=(16, 2))

#     # Caso 1: None
#     if safety_margin is None:
#         ax.text(0.5, 0.5, "Safety Margin N/A",
#                 ha="center", va="center", fontsize=12)
#         ax.set_axis_off()
#         return fig

#     # Caso 2: escalar (float ou int)
#     if isinstance(safety_margin, (int, float, np.floating)):
#         data = np.array([[safety_margin]])

#         sns.heatmap(
#             data,
#             cmap="coolwarm",
#             center=0,
#             annot=True,
#             fmt=".2f",
#             cbar=True,
#             ax=ax
#         )

#         ax.set_title("Safety Margin (Global)")
#         ax.set_xticks([])
#         ax.set_yticks([])

#         return fig

#     # Caso 3: vetor por slot
#     safety_margin = np.asarray(safety_margin, dtype=float)

#     if safety_margin.size == 0:
#         ax.text(0.5, 0.5, "Safety Margin N/A",
#                 ha="center", va="center", fontsize=12)
#         ax.set_axis_off()
#         return fig

#     data = safety_margin.reshape(1, -1)

#     sns.heatmap(
#         data,
#         cmap="coolwarm",
#         center=0,
#         cbar=True,
#         ax=ax
#     )

#     ax.set_title("Safety Margin por Slot (Motoristas − Demanda)")
#     ax.set_xlabel("Slot")
#     ax.set_yticks([])

#     return fig



# # ------------------------------------------------
# # Heatmap – Coverage (escala global)
# # ------------------------------------------------
# def plot_heatmap_coverage(coverage):
#     fig, ax = plt.subplots(figsize=(4, 2))

#     if coverage is None or np.isnan(coverage):
#         ax.text(0.5, 0.5, "Coverage N/A",
#                 ha="center", va="center", fontsize=12)
#         ax.set_axis_off()
#         return fig

#     data = np.array([[coverage]])

#     sns.heatmap(
#         data,
#         cmap="Greens",
#         center=1.0,
#         annot=True,
#         fmt=".3f",
#         cbar=True,
#         ax=ax
#     )

#     ax.set_title("Global Coverage Score")
#     ax.set_xticks([])
#     ax.set_yticks([])

#     return fig


# # ------------------------------------------------
# # Radar Chart – KPIs agregados
# # ------------------------------------------------
# def plot_kpi_radar(kpis_dict):
#     import numpy as np

#     # Remove KPIs inválidos
#     labels = []
#     values = []

#     for k, v in kpis_dict.items():
#         if v is not None and not np.isnan(v):
#             labels.append(k)
#             values.append(v)

#     if len(values) < 3:
#         fig, ax = plt.subplots()
#         ax.text(0.5, 0.5, "KPIs insuficientes",
#                 ha="center", va="center")
#         ax.set_axis_off()
#         return fig

#     values += values[:1]
#     angles = np.linspace(0, 2 * np.pi, len(values))

#     fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
#     ax.plot(angles, values, linewidth=2)
#     ax.fill(angles, values, alpha=0.3)

#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=8)
#     ax.set_title("Comparação dos KPIs")

#     return fig


# solver/charts.py
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    # Theme configuration - PROFESSIONAL SOBER
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Roboto", "Helvetica", "Arial"],
        "axes.edgecolor": "#E2E8F0",
        "axes.linewidth": 1.0,
        "grid.color": "#F1F5F9",
        "text.color": "#1E293B",
        "axes.labelcolor": "#475569",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
except Exception:
    sns = None

from src.core.i18n import t



def _to_1d_numeric(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.array([float(x)], dtype=float)
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return np.array([float(arr)], dtype=float)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr
    return None


def plot_heatmap_coverage(coverage: Any):
    """
    Heatmap de coverage por slot.
    Retorna fig ou None.
    """
    cov = _to_1d_numeric(coverage)
    if cov is None or cov.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(16, 2))

    data = cov.reshape(1, -1)  # (1, n)

    if sns is not None:
        sns.heatmap(data, cmap="YlGnBu", center=1, cbar=True, ax=ax)
    else:
        ax.imshow(data, aspect="auto")
        ax.set_title(f"{t('chart_heatmap')} (fallback)")

    ax.set_title(t('chart_heatmap'))
    ax.set_xlabel(t('axis_slot'))
    ax.set_ylabel("")
    return fig


def plot_heatmap_safety(safety_margin: Any):
    """
    Heatmap de margem (workers - demanda) por slot.
    Retorna fig ou None.
    """
    sm = _to_1d_numeric(safety_margin)
    if sm is None or sm.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(16, 2))

    data = sm.reshape(1, -1)  # (1, n)

    if sns is not None:
        sns.heatmap(data, cmap="RdYlBu_r", center=0, cbar=True, ax=ax)
    else:
        ax.imshow(data, aspect="auto")
        ax.set_title(f"{t('chart_safety')} (fallback)")

    ax.set_title(t('chart_safety'))
    ax.set_xlabel(t('axis_slot'))
    ax.set_ylabel("")
    return fig


def plot_kpi_radar(kpis_dict: Dict[str, Any]):
    """
    Radar simples e defensivo: ignora valores None/não numéricos.
    Retorna fig ou None.
    """
    labels = []
    values = []
    FIG_HEIGHT = 4.5

    for k, v in (kpis_dict or {}).items():
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
            labels.append(k)
            values.append(float(v))

    if len(values) < 3:
        return None  # radar não faz sentido com poucos pontos

    # fecha o polígono
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=True)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, FIG_HEIGHT))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title(t('chart_kpi'))
    return fig
