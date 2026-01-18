import streamlit as st
import pulp
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile  # Para criar arquivos temporários

# Função de otimização (baseada no código anterior)
def solve_shift_schedule(need):
    num_periods = len(need)
    prob = pulp.LpProblem("24_hour_shift_scheduling", pulp.LpMinimize)
    
    # Variáveis de decisão
    X = pulp.LpVariable.dicts("X", range(num_periods), cat="Binary")
    
    # Função objetivo: minimizar o número total de trabalhadores
    prob += pulp.lpSum(X[d] for d in range(num_periods)), "Minimize_Workers"
    
    # Matriz de restrições
    constraint_matrix = np.zeros((num_periods, num_periods), dtype=int)
    
    # Restrições de cobertura de necessidade em cada período
    for i in range(num_periods):
        prob += (
            pulp.lpSum(
                X[j] for j in range(num_periods) 
                if (0 <= (i - j) % num_periods < 18)  # Primeiro bloco de 18 períodos
                or (19 <= (i - j) % num_periods < 23)  # Bloco de 4 períodos de descanso
                or (25 <= (i - j) % num_periods < 43)  # Segundo bloco de 18 períodos
            ) >= need[i]
        ), f"Coverage_period_{i}"
        
        # Atualizar matriz de restrições
        for j in range(num_periods):
            if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
                constraint_matrix[i, j] = 1

    # Criar um arquivo temporário para capturar a saída do solver
    with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as temp_log_file:
        log_path = temp_log_file.name
        
    # Resolver o problema, salvando a saída em um arquivo de log
    solver = pulp.PULP_CBC_CMD(msg=True, logPath=log_path)
    prob.solve(solver)

    # Extrair resultados
    workers_schedule = [int(pulp.value(X[d])) for d in range(num_periods)]
    total_workers = pulp.value(prob.objective)
    
    # Calcular densidade
    density = np.sum(constraint_matrix) / (num_periods ** 2)
    
    # Ler a saída do solver do arquivo de log
    with open(log_path, 'r') as f:
        solver_output_str = f.read()

    # Dividir a saída do solver em partes
    # Parte sempre visível
    visible_output = "\n".join(line for line in solver_output_str.splitlines() if "Result - Optimal solution found" in line or line.startswith("Objective value:") or line.startswith("Enumerated nodes:") or line.startswith("Total iterations:") or line.startswith("Time (CPU seconds):") or line.startswith("Total time (CPU seconds):"))

    # Parte colapsável
    detailed_output = "\n".join(line for line in solver_output_str.splitlines() if "Result - Optimal solution found" not in line and not (line.startswith("Objective value:") or line.startswith("Enumerated nodes:") or line.startswith("Total iterations:") or line.startswith("Time (CPU seconds):") or line.startswith("Total time (CPU seconds):")))

    # Determinar o tipo do modelo
    model_class = type(prob)   # O seu modelo é um Mixed Integer Linear Program
    
    # Exibindo o status da solução
    st.write("Status da Solução:", LpStatus[prob.status])

    # Função para determinar o tipo de modelo
    def tipo_modelo(prob):
        if all(v.cat == 'Continuous' for v in prob.variables()):
         return "LP (Programação Linear)"
        elif any(v.cat in ['Integer', 'Binary'] for v in prob.variables()):
            if all(v.cat in ['Integer', 'Binary'] for v in prob.variables()):
                return "MILP (Programação Linear Inteira Mista)"
            else:
                return "MIP (Programação Inteira Mista)"
        else:
            return "Modelo Desconhecido"

    # Exibindo o tipo do modelo
    st.write("Tipo do Modelo:", tipo_modelo(prob))

    # Exibindo o valor da função objetivo
    st.write(f"Valor da Função Objetivo: {prob.objective.value()}")

    
    return total_workers, workers_schedule, constraint_matrix, density, visible_output, detailed_output, model_class

# Interface do Streamlit
st.title("24-Hour Shift Scheduler")

st.write("""
Este aplicativo otimiza o agendamento de turnos de trabalhadores para cobrir a necessidade de cada período de 15 minutos ao longo de 24 horas.
Insira a demanda de trabalhadores para cada período.
""")

# Entrada do usuário: uma lista de números separados por vírgula para representar a demanda por trabalhadores
default_need = "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"
need_input = st.text_area("Demanda de trabalhadores (separado por vírgulas para cada período de 15 minutos):", default_need)

# Converter a entrada do usuário em uma lista de inteiros
try:
    need = list(map(int, need_input.split(',')))

    # Verificar se a entrada tem exatamente 96 períodos
    if len(need) != 96:
        st.error("A entrada deve ter exatamente 96 valores (1 para cada período de 15 minutos).")
    else:
        # Resolver o problema de otimização
        total_workers, workers_schedule, constraint_matrix, density, visible_output, detailed_output, model_class = solve_shift_schedule(need)

        # Exibir os resultados
        st.subheader("Resultados")
        st.write(f"**Total de trabalhadores necessários:** {total_workers}")
        st.write("**Escalonamento dos trabalhadores (1 significa que um trabalhador começa neste período):**")
        st.write(workers_schedule)

        # Visualização dos dados
        st.bar_chart(workers_schedule)

        # Exibir a matriz de restrições
        st.subheader("Matriz de Restrições")
        st.write("A matriz de restrições mostra quais trabalhadores podem cobrir cada período.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(constraint_matrix, cmap="Blues", cbar=True, ax=ax)
        ax.set_title("Matriz de Restrições (1 = Cobertura do Período, 0 = Não Coberto)")
        ax.set_xlabel("Período")
        ax.set_ylabel("Trabalhador")
        st.pyplot(fig)

        # Exibir a densidade da matriz de restrições
        st.subheader("Densidade da Matriz de Restrições")
        st.write(f"A densidade da matriz de restrições é: **{density:.4f}**")
        st.write("Densidade é a proporção de restrições ativas em relação ao total possível de restrições.")

        # Exibir a parte visível da saída do solver
        st.subheader("Saída do Solver")
        st.text(visible_output)

        # Exibir a parte detalhada da saída do solver em um expander
        with st.expander("Detalhes Adicionais do Solver", expanded=False):
            st.text(detailed_output)

        # Exibir o tipo do modelo
        st.subheader("Tipo do Modelo")
        st.write(f"**Modelo:** {model_class}")

except ValueError:
    st.error("Entrada inválida. Por favor, insira números inteiros separados por vírgulas.")
