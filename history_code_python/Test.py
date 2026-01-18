import streamlit as st
import pandas as pd
import numpy as np

# Gerar dados de exemplo
num_periods = 10
num_workers = 5
data = np.random.randint(1, 10, size=(num_periods, num_workers))

# Preparar os dados para o gráfico de barras
constraint_data = pd.DataFrame(
    data,
    columns=[f'Worker {i+1}' for i in range(num_workers)],
    index=[f'Period {i+1}' for i in range(num_periods)]
)

# Exibir a densidade
initialDensityMatrix = np.mean(data)  # Exemplo de densidade calculada
st.write(f"Density: {initialDensityMatrix:.4f}")

# Exibir o gráfico de barras
with st.expander("Initial Constraint Matrix", expanded=True):
    # Normalizar os dados para simular a intensidade (semelhante a um heatmap)
    norm_data = (constraint_data - constraint_data.min()) / (constraint_data.max() - constraint_data.min())
    
    # Multiplicar por um valor maior para ajustar a intensidade visual (simulando a paleta de cores de um heatmap)
    norm_data *= 100  # Ajustando para escala de 0 a 100
    
    # Plotar os dados normalizados com st.bar_chart
    st.bar_chart(norm_data)
