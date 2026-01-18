import streamlit as st
import pulp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile  # Para criar arquivos temporários

# Função de otimização (baseada no código anterior)
def solve_shift_schedule(need, variable_type, solver_choice):
    num_periods = len(need)
    prob = pulp.LpProblem("Shift_Scheduling", pulp.LpMinimize)
    
    # Mapeando o tipo de variável escolhido pelo usuário
    if variable_type == "Contínua":
        cat = pulp.LpContinuous
    elif variable_type == "Binária":
        cat = pulp.LpBinary
    elif variable_type == "Inteira":
        cat = pulp.LpInteger
    else:
        raise ValueError("Tipo de variável inválido.")
    
    # Variáveis de decisão
    X = pulp.LpVariable.dicts("X", range(num_periods), cat=cat)
    
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
        
    # Selecionar o solver com base na escolha do usuário
    if solver_choice == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=True, logPath=log_path)
    elif solver_choice == "GLPK":
        solver = pulp.GLPK_CMD(msg=True, options=["--log", log_path])
    elif solver_choice == "CPLEX":
        solver = pulp.CPLEX_CMD(msg=True, logPath=log_path)
    elif solver_choice == "Gurobi":
        solver = pulp.GUROBI_CMD(msg=True, logPath=log_path)
    else:
        st.error("Solver inválido. Selecione um solver disponível.")
        return None, None, None, None, None, None, None

    # Resolver o problema
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
    visible_output = "\n".join(line for line in solver_output_str.splitlines() if "Result - Optimal solution found" in line or line.startswith("Objective value:") or line.startswith("Enumerated nodes:") or line.startswith("Total iterations:") or line.startswith("Time (CPU seconds):") or line.startswith("Total time (CPU seconds):"))
    
    detailed_output = "\n".join(line for line in solver_output_str.splitlines() if "Result - Optimal solution found" not in line and not (line.startswith("Objective value:") or line.startswith("Enumerated nodes:") or line.startswith("Total iterations:") or line.startswith("Time (CPU seconds):") or line.startswith("Total time (CPU seconds):")))

    # Determinar o tipo do modelo
    model_class = "MILP"  # O seu modelo é um Mixed Integer Linear Program
    
    return total_workers, workers_schedule, constraint_matrix, density, visible_output, detailed_output, model_class

# Interface do Streamlit
st.title("24-Hour Shift Scheduler")

st.write(""" 
Este aplicativo otimiza o agendamento de turnos de trabalhadores para cobrir a necessidade de cada período de tempo ao longo de um dia. 
Insira o número total de horas e o intervalo em minutos.
""")

# Entradas do usuário para horas e períodos
total_hours = st.number_input("Quantidade total de horas:", min_value=1, value=24)
period_minutes = st.number_input("Duração de cada período (em minutos):", min_value=1, value=15)

# Calcular o número total de períodos
num_periods = (total_hours * 60) // period_minutes

# Gerar demanda aleatória entre 0 e 10 para cada período
default_need = np.random.randint(0, 11, size=num_periods).tolist()  # Gera uma lista com valores aleatórios de 0 a 10
need_input = st.text_area(f"Demanda de trabalhadores (separado por vírgulas para cada período de {period_minutes} minutos):", ', '.join(map(str, default_need)))

# Exibir a quantidade de elementos em default_need
st.write(f"**Número de elementos em 'default_need': {len(default_need)}**")

# Seleção do tipo de variável
variable_type = st.selectbox("Escolha o tipo de variável:", ["Contínua", "Binária", "Inteira"])

# Seleção do solver
solver_choice = st.selectbox("Escolha o solver:", ["CBC", "GLPK", "CPLEX", "Gurobi"])

# Botão para executar a otimização
if st.button("Executar Otimização"):
    # Converter a entrada do usuário em uma lista de inteiros
    try:
        need = list(map(int, need_input.split(',')))

        # Verificar se a entrada tem o número correto de períodos
        if len(need) != num_periods:
            st.error(f"A entrada deve ter exatamente {num_periods} valores (1 para cada período de {period_minutes} minutos).")
        else:
            # Resolver o problema de otimização
            total_workers, workers_schedule, constraint_matrix, density, visible_output, detailed_output, model_class = solve_shift_schedule(need, variable_type, solver_choice)

            # Exibir os resultados
            if total_workers is not None:
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
