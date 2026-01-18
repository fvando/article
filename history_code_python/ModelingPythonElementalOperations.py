import streamlit as st
import pulp
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile  # Para criar arquivos temporários
import time  # Para medir o tempo de execução


# Função para determinar o tipo de modelo
def tipo_modelo(prob):
    has_integer = any(v.cat in ['Integer', 'Binary'] for v in prob.variables())
    has_continuous = any(v.cat == 'Continuous' for v in prob.variables())
    
    # Checar se existe uma expressão não linear
    has_nonlinear = any('nonlinear' in str(const) for const in prob.constraints.values())

    # Verificar se existem expressões quadráticas
    has_quadratic = any('quadratic' in str(obj) for obj in [prob.objective] + list(prob.constraints.values()))

    # Determinando o tipo do modelo
    if has_nonlinear:
        if has_integer:
            return "MINLP (Integer Nonlinear Program)"
        return "NLP (Nonlinear Program)"

    if has_quadratic:
        if has_integer:
            return "MIQP (Mixed Integer Quadratic Program)"
        return "QP (Quadratic Program)"

    if has_integer:
        if all(v.cat in ['Integer', 'Binary'] for v in prob.variables()):
            return "PILP (Pure Integer Linear Program)"
        return "MILP (Mixed Integer Linear Program)"

    if all(v.cat == 'Continuous' for v in prob.variables()):
        return "LP (Linear Program)"

    return "Modelo Desconhecido"

# Função para aplicar operações elementares
def apply_elementary_operations(constraint_matrix):
    st.subheader("Operações Elementares nas Restrições")
    
    swap_rows = st.checkbox("Troca de Equações")
    multiply_row = st.checkbox("Multiplicação por Constante")
    add_multiple_rows = st.checkbox("Somar Múltiplo de uma Equação a Outra")
    
    if swap_rows:
        row1 = st.number_input("Escolha a linha 1 para trocar:", min_value=0, max_value=len(constraint_matrix)-1)
        row2 = st.number_input("Escolha a linha 2 para trocar:", min_value=0, max_value=len(constraint_matrix)-1)
        if st.button("Aplicar Troca"):
            constraint_matrix[[row1, row2]] = constraint_matrix[[row2, row1]]
            st.success(f"Linhas {row1} e {row2} trocadas com sucesso!")
    
    if multiply_row:
        row = st.number_input("Escolha a linha para multiplicar:", min_value=0, max_value=len(constraint_matrix)-1)
        constant = st.number_input("Escolha a constante para multiplicar:", value=1.0)
        if constant != 0 and st.button("Aplicar Multiplicação"):
            constraint_matrix = constraint_matrix.astype(np.float64)
            constraint_matrix[row] *= constant
            st.success(f"Linha {row} multiplicada por {constant} com sucesso!")
        else:
            st.warning("A constante de multiplicação não pode ser zero!")

    if add_multiple_rows:
        row1 = st.number_input("Escolha a linha base para somar:", min_value=0, max_value=len(constraint_matrix)-1, key="row1_sum")
        row2 = st.number_input("Escolha a linha que vai receber o múltiplo:", min_value=0, max_value=len(constraint_matrix)-1, key="row2_sum")
        multiple = st.number_input("Escolha o múltiplo para somar:", value=1.0, key="multiple_sum")
        if st.button("Aplicar Soma"):
            constraint_matrix = constraint_matrix.astype(np.float64)
            constraint_matrix[row2] += multiple * constraint_matrix[row1]
            st.success(f"Múltiplo da linha {row1} somado à linha {row2} com sucesso!")
    
    # Verifique o resultado final da matriz após as operações
    #st.write("Matriz de restrições após as operações:")
    with st.expander("Matriz de restrições após as operações:", expanded=False):
        st.write(constraint_matrix)

    return constraint_matrix

# Função de otimização (baseada no código anterior)
def solve_shift_schedule(need, variable_type, solver_choice, constraint_matrix):
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

    # Preencher a matriz de restrições de acordo com os períodos cobertos
    #    for j in range(num_periods):
    #        if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
    #            constraint_matrix[i, j] = 1  # Atualizando a matriz de restrições com os valores de cobertura
  
    # Exibir a matriz de restrições inicial
    #st.subheader("Matriz de Restrições Inicial")
    with st.expander("Matriz de Restrições Inicial", expanded=False):
        st.write(constraint_matrix)
    
    
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

    
    # Exibindo o status da solução
    st.write("Status da Solução:", LpStatus[prob.status])

    # Exibindo o tipo do modelo
    st.write("Tipo do Modelo:", tipo_modelo(prob))

    # Determinar o tipo do modelo
    model_class = tipo_modelo(prob)  #"MILP"  # O seu modelo é um Mixed Integer Linear Program
    
    
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

# Matriz de restrições
constraint_matrix = np.zeros((num_periods, num_periods), dtype=int)

# Preencher a matriz de restrições de acordo com os períodos cobertos
need = list(map(int, need_input.split(',')))
num_periods = len(need)
for i in range(num_periods):
    for j in range(num_periods):
        if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
            constraint_matrix[i, j] = 1  # Atualizando a matriz de restrições com os valores de cobertura

# Aplicar operações elementares à matriz de restrições **antes** da execução
constraint_matrix = apply_elementary_operations(constraint_matrix)

# Botão para executar a otimização
#if st.button("Executar Otimização"):
# Converter a entrada do usuário em uma lista de inteiros
try:
    need = list(map(int, need_input.split(',')))
    #st.write("Matriz de restrições antes da otimização:")
    with st.expander("Matriz de restrições antes da otimização:", expanded=False):
        st.write(constraint_matrix)  
    
    # Verificar se a entrada tem o número correto de períodos
    if len(need) != num_periods:
        st.error(f"A entrada deve ter exatamente {num_periods} valores (1 para cada período de {period_minutes} minutos).")
    else:
        # Resolver o problema de otimização
        total_workers, workers_schedule, constraint_matrix, density, visible_output, detailed_output, model_class = solve_shift_schedule(need, variable_type, solver_choice, constraint_matrix)
        # Exibir os resultados
        if total_workers is not None:
            st.subheader("Resultados")
            st.write(f"**Total de trabalhadores necessários:** {total_workers}")
            
            # Adicionando uma seção colapsável para o escalonamento
            with st.expander("Escalonamento dos Trabalhadores", expanded=False):
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