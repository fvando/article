import matplotlib.pyplot as plt
from fpdf import FPDF
import streamlit as st

def validate_input(num_slots, demand):
    """Valida se o número de slots e a demanda são compatíveis."""
    return len(demand) == num_slots

def plot_allocation(solution, needs, time_slots):
    """Plota a alocação de motoristas em relação às necessidades."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(time_slots), needs, label="Necessidades", color='blue', alpha=0.7)
    ax.bar(range(time_slots), solution, label="Motoristas Alocados", color='orange', alpha=0.7)
    ax.set_xlabel("Slots de Tempo")
    ax.set_ylabel("Número de Motoristas")
    ax.set_title("Alocação de Motoristas por Slot")
    ax.legend()
    plt.grid(True)
    st.pyplot(fig)

def generate_pdf_report(solution, needs, total_drivers):
    """Gera um relatório PDF com os resultados."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Alocação de Motoristas", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Total de motoristas necessários: {total_drivers}", ln=True)
    pdf.cell(200, 10, txt="Detalhes da Alocação:", ln=True)

    for i, (need, sol) in enumerate(zip(needs, solution)):
        pdf.cell(200, 10, txt=f"Slot {i+1}: Necessidade = {need}, Alocado = {sol}", ln=True)

    pdf.output("report.pdf")
