import streamlit as st
from ortools.linear_solver import pywraplp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tempfile  # Para criar arquivos temporários
import io
import sys
import json
import os

# Habilitar a largura total da página
st.set_page_config(layout="wide")

# Caminho do arquivo onde o estado será salvo
FILE_PATH = "matrix_state.json"

# Função para salvar o estado atual da matriz em um arquivo JSON
def save_state(constraints_coefficients, need):
    with open(FILE_PATH, "w") as file:
        json.dump({
            "constraints_coefficients": constraints_coefficients,
            "need": need
        }, file)
        
    # Função para carregar o estado da matriz do arquivo JSON, se ele existir
def load_state():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            state = json.load(file)
            return state["constraints_coefficients"], state["need"]
    else:
        # Retorna o estado inicial caso o arquivo não exista
        return [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ], [1, 8, 3, 4]  # Valores iniciais de `constraints_coefficients` e `need`

# Carregar o estado inicial ou salvo
constraints_coefficients, need = load_state()
    
# Definindo as restrições
restrictions = [
    {
        "Descrição": "Limite diário de condução",
        "Fórmula": r"$\sum_{p \in \text{dia}} X[p] \leq 36$",
        "Detalhes": "Pode ser 40 períodos (10 horas) duas vezes por semana.",
        "Key": "limite_diario"
    },
    {
        "Descrição": "Limite semanal de condução",
        "Fórmula": r"$\sum_{p \in \text{semana}} X[p] \leq 224$",
        "Detalhes": "Total de períodos de trabalho durante uma semana não deve ultrapassar 224 períodos.",
        "Key": "limite_semanal"
    },
    {
        "Descrição": "Limite quinzenal de condução",
        "Fórmula": r"$\sum_{p \in \text{quinzena}} X[p] \leq 360$",
        "Detalhes": "Total de períodos de trabalho em duas semanas não deve ultrapassar 360 períodos.",
        "Key": "limite_quinzenal"
    },
    {
        "Descrição": "Repouso diário mínimo",
        "Fórmula": r"$\text{Repouso} \geq 44$",
        "Detalhes": "O motorista deve descansar pelo menos 44 períodos (11 horas) todos os dias.",
        "Key": "repouso_diario_minimo"
    },
    {
        "Descrição": "Repouso diário reduzido",
        "Fórmula": r"$\text{Repouso} \geq 36 \text{ (máx. 3 vezes em 14 dias)}$",
        "Detalhes": "O repouso pode ser reduzido para 36 períodos (9 horas), mas não mais do que 3 vezes em 14 dias.",
        "Key": "repouso_diario_reduzido"
    },
    {
        "Descrição": "Repouso semanal",
        "Fórmula": r"$\text{Repouso semanal} \geq 180$",
        "Detalhes": "O motorista deve ter um período de repouso de 180 períodos (45 horas) toda semana.",
        "Key": "repouso_semanal"
    },
    {
        "Descrição": "Repouso quinzenal",
        "Fórmula": r"$\text{Repouso quinzenal} \geq 96$",
        "Detalhes": "O motorista deve ter um período de repouso de 96 períodos (24 horas) a cada duas semanas.",
        "Key": "repouso_quinzenal"
    },
    {
        "Descrição": "Descanso após dias de trabalho",
        "Fórmula": r"$\text{6 dias de trabalho} \Rightarrow \text{Repouso}$",
        "Detalhes": "O repouso semanal deve ser gozado após seis dias de trabalho consecutivos.",
        "Key": "descanso_apos_trabalho"
    },
    {
        "Descrição": "Pausa de 45 minutos após 4,5 horas de condução",
        "Fórmula": r"$\text{Pausa} \geq 3$",
        "Detalhes": "Após no máximo 18 períodos (4,5 horas), deve haver uma pausa de 3 períodos (45 minutos).",
        "Key": "pausa_45_minutos"
    },
    {
        "Descrição": "Divisão da pausa",
        "Fórmula": r"$\text{Pausa} = \text{1 período de 15 minutos} + \text{1 período de 30 minutos}$",
        "Detalhes": "A pausa de 3 períodos pode ser feita como 15 minutos seguidos de 30 minutos.",
        "Key": "divisao_pausa"
    }
]

# Função para formatar a saída do modelo
def format_lp_output(num_vars, num_restricoes, rhs_values):
    """
    Formata a saída de um problema de programação linear com base nas variáveis, restrições, 
    e nos valores rhs específicos para cada restrição.
    
    Parâmetros:
        - variables: lista das variáveis de decisão.
        - constraints: lista de listas, onde cada sublista representa os coeficientes das variáveis em uma restrição.
        - rhs_values: lista de valores rhs específicos para cada restrição.
    
    Retorno:
        - Uma string formatada representando o problema de programação linear.
    """
    
       # Verificação para evitar divisão por zero
    #if num_restricoes == 0:
    #    num_restricoes = 1  # Definimos pelo menos 1 restrição para evitar erro
    
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
    
    
    #output += "MODEL:\n\n"
    #output += " [_1] MIN= " + " + ".join(f"X_{i+1}" for i in range(num_vars)) + ";\n\n"

    # Variáveis
    #for j in range(num_vars):
    #    restricao = " + ".join(f"X_{i+1}" for i in range(j + 1)) + " + " \
    #                + " + ".join(f"X_{i+1}" for i in range(num_vars))
    #    output += f" [_{j+2}] {restricao} >= {rhs_values[j]};\n\n"
    
    # Restrictions
    #for z in range(num_restricoes):
    #    restricao2 = " + ".join(f"X_{i+1}" for i in range(z + 1)) + " + " \
    #                + " + ".join(f"X_{i+1}" for i in range(num_restricoes))
    #    output += f" [_{z+2}] {restricao2} >= {rhs_values[z]};\n\n"


    # Variáveis inteiras
    #output += " " + " ".join(f"@GIN(X_{i+1});" for i in range(num_vars)) + "\n\n"
    #output += "END"

    return output

# Função para exibir o modelo de otimização
def generate_model(num_periods, variable_type, selected_restrictions, need):
    
    # Marcar uma das restrições como default (exemplo: a primeira)
    #default_key = restrictions[0]["Key"] if restrictions else None  # Pega a chave da primeira restrição
    #if default_key:
    #    selected_restrictions[default_key]  # Marca como selecionada
 
    st.subheader("Modelo de Otimização")
    st.write("**Variáveis de decisão:**")
    st.write(f"Tipo de variável: {variable_type}")

    st.write("**Função objetivo:**")
    st.write("Minimizar o número total de trabalhadores necessários para cobrir a demanda.")

    st.write("**Restrições:**")
    for restricao in restrictions:
        if selected_restrictions[restricao["Key"]]:
            st.write(f"{restricao['Descrição']}: {restricao['Fórmula']}")

    st.write("**Demanda de trabalhadores por período:**")
    with st.expander("Trabalhadores", expanded=False):
        st.write(need)
    
# Função para determinar o tipo de modelo
def tipo_modelo(solver):
    return "Modelo Linear ou Inteiro"

# Função para aplicar operações elementares
def apply_elementary_operations(constraint_matrix):
    #with st.expander("Matriz de restrições pós operações:", expanded=False):
    st.subheader("Operações Elementares nas Restrições")
    with st.expander("Escalonamento dos Trabalhadores", expanded=False):
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
        
        #with st.expander("Matriz de restrições após as operações:", expanded=False):
        st.write(constraint_matrix)

    return constraint_matrix

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
        
# Função de otimização (baseada no código anterior)
def solve_shift_schedule(solverParamType, need, variable_type, initualconstraintsCoefficients, constraint_matrix, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None):
    num_periods = len(need)
    initualconstraintsCoefficients[:] = []
    
    # Criar o solver
    solver = pywraplp.Solver.CreateSolver(solverParamType)  # Usar 'GLOP' para LP ou 'SCIP' para MIP
   
    # Variáveis de decisão
    if variable_type == "Contínua":
        X = [solver.NumVar(0, solver.infinity(), f'X[{d}]') for d in range(num_periods)]
    elif variable_type == "Binária":
        X = [solver.BoolVar(f'X[{d}]') for d in range(num_periods)]
    elif variable_type == "Inteira":
        X = [solver.IntVar(0, solver.infinity(), f'X[{d}]') for d in range(num_periods)]
    else:
        raise ValueError("Tipo de variável inválido.")

    # Função objetivo: minimizar o número total de trabalhadores
    solver.Minimize(solver.Sum(X[d] for d in range(num_periods)))

    for i in range(num_periods):
        constraint = solver.Add(solver.Sum(X[j] for j in range(num_periods) 
                               if (0 <= (i - j) % num_periods < 18)  
                               or (19 <= (i - j) % num_periods < 23)  
                               or (25 <= (i - j) % num_periods < 43) ) >= need[i])
        constraints.append(constraint)
    
    #constraints = []
    #constraints_coefficients = []
    
    # Restrições de cobertura de necessidade em cada período
    for i in range(num_periods):
        coeffs = []
        for j in range(num_periods):
            if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
                coeffs.append(1)
            else:
                coeffs.append(0)  # Ou outro valor conforme necessário
        constraints_coefficients.append(coeffs)    

    # Para depuração: Exibir a matriz de coeficientes como uma tabela
    #df = pd.DataFrame(constraints_coefficients)
    #st.write("Matriz de Coeficientes:")
    #st.table(df)

    initualconstraintsCoefficients =  constraints_coefficients
    
    # Aplicar operações elementares, se selecionadas
    #A troca de linhas não reduz diretamente o fill-in, mas pode ajudar na preparação de operações de eliminação.
    # Exemplo de troca de linhas na matriz de coeficientes
    if swap_rows is not None:
        # Exibir o sistema linear antes da troca de linhas
        st.write("Sistema Linear antes da troca de linhas:")
        display_system(constraints_coefficients,need)
        row1, row2 = swap_rows
        if (0 <= row1 < len(constraints_coefficients)) and (0 <= row2 < len(constraints_coefficients)):
            # Troca de linhas na matriz de coeficientes
            constraints_coefficients[row1], constraints_coefficients[row2] = constraints_coefficients[row2], constraints_coefficients[row1]
            # Troca correspondente no vetor de resultados
            need[row1], need[row2] = need[row2], need[row1]
            # Atualização de restrições no solver
            constraints[row1], constraints[row2] = constraints[row2], constraints[row1]
            print(f"Linhas {row1} e {row2} trocadas com sucesso!")

        # Exibir o sistema linear após a troca de linhas
        st.write("Sistema Linear após a troca de linhas:")
        display_system(constraints_coefficients,need)

        # Para depuração: Exibir a matriz de coeficientes como uma tabela
        df = pd.DataFrame(constraints_coefficients)
        st.write("Matriz de Coeficientes:")
        # st.table(df)

    # A multiplicação de uma linha por uma constante é útil se for usada para cancelar ou simplificar valores, mas pode aumentar o fill-in se não for cuidadosamente controlada.
    # Exemplo de multiplicação de uma linha por uma constante
    if multiply_row is not None:
        # Exibir o sistema linear antes da troca de linhas
        st.write("Sistema Linear antes da multiplicação de uma linha por uma constante:")
        display_system(constraints_coefficients,need)
        row, constant = multiply_row
        if 0 <= row < len(constraints_coefficients):
            if constant != 0:
                # Multiplicação dos coeficientes por uma constante
                constraints_coefficients[row] = [c * constant for c in constraints_coefficients[row]]
                # Atualização da restrição correspondente no solver
                new_expr = solver.Sum(constraints_coefficients[row][j] * X[j] for j in range(len(X)))
                constraints[row] = solver.Add(new_expr >= constraints[row].lb() * constant)
                print(f"Linha {row} multiplicada por {constant} com sucesso!")

        st.write("Sistema Linear após da multiplicação de uma linha por uma constante:")
        display_system(constraints_coefficients,need)

        # Para depuração: Exibir a matriz de coeficientes como uma tabela
        df = pd.DataFrame(constraints_coefficients)
        st.write("Matriz de Coeficientes:")
        # st.table(df)


    if add_multiple_rows is not None:
        # Exibir o sistema linear antes da troca de linhas
        st.write("Sistema Linear antes da adição de múltiplo de uma linha à outra linha:")
        display_system(constraints_coefficients, need)
        row1, row2, multiple = add_multiple_rows
        if 0 <= row1 < len(constraints_coefficients) and 0 <= row2 < len(constraints_coefficients):
            
            # Adicionar múltiplo da linha 1 à linha 2
            constraints_coefficients[row2] = [
                constraints_coefficients[row2][j] + multiple * constraints_coefficients[row1][j]
                for j in range(len(X))
            ]
            
            # Atualizar o lado direito da equação para a linha row2 no vetor need
            need[row2] = need[row2] + multiple * need[row1]
            
            # Atualizar a restrição no solver com os novos valores
            new_expr = solver.Sum(constraints_coefficients[row2][j] * X[j] for j in range(len(X)))
            constraints[row2] = solver.Add(new_expr >= need[row2])

            print(f"Múltiplo da linha {row1} somado à linha {row2} com sucesso!")

        # Salvar o estado atualizado após a operação
        save_state(constraints_coefficients, need)

        # Exibir o sistema linear após a troca de linhas
        st.write("Sistema Linear após da Sistema Linear após da adição de múltiplo de uma linha à outra linha:")
        display_system(constraints_coefficients, need)

        # Para depuração: Exibir a matriz de coeficientes como uma tabela
        df = pd.DataFrame(constraints_coefficients)
        st.write("Matriz de Coeficientes:")
        # st.table(df)

    # Adicionar as outras restrições conforme o seu código original...
    # 1. Limite diário de condução (9 horas ou 10 horas duas vezes por semana)
    if selected_restrictions["limite_diario"]:    
        for day in range(num_periods // 96):  # 96 períodos de 15 minutos em um dia
          constraint =  solver.Add(solver.Sum(X[day * 96 + p] for p in range(96)) <= 36)  # Limite normal de 9 horas (36 períodos)
          constraints.append(constraint)

    # Adiciona exceção para 10 horas de condução (40 períodos), no máximo 2 vezes por semana
    if selected_restrictions["limite_semanal"]:    
        extended_hours_days = 2  # Número de dias que pode trabalhar 10 horas
        extended_hours_vars = []
        for day in range(num_periods // 96):  # 96 períodos por dia
            extended_hours = solver.BoolVar(f'extended_hours[{day}]')
            extended_hours_vars.append(extended_hours)
            constraint = solver.Add(solver.Sum(X[day * 96 + p] for p in range(96)) <= 40 * extended_hours + 36 * (1 - extended_hours))
            constraints.append(constraint)
        constraint = solver.Add(solver.Sum(extended_hours_vars) <= extended_hours_days)
        constraints.append(constraint)

    # 2. Limite semanal de condução: 56 horas por semana
    if selected_restrictions["limite_semanal"]:    
        for week in range(num_periods // (96 * 7)):  # Cada semana tem 96 * 7 períodos de 15 minutos
            constraint = solver.Add(solver.Sum(X[week * 96 * 7 + p] for p in range(96 * 7)) <= 224)
            constraints.append(constraint)

    # 3. Limite quinzenal de condução: 90 horas a cada 14 dias
    if selected_restrictions["limite_quinzenal"]:    
        for period in range(num_periods // (96 * 14)):  # 14 dias com 96 períodos por dia
            constraint = solver.Add(solver.Sum(X[period * 96 * 14 + p] for p in range(96 * 14)) <= 360)
            constraints.append(constraint)
            
    # 4. Repouso diário mínimo de 11 horas (44 períodos)
    if selected_restrictions["repouso_diario_minimo"]:    
        daily_rest_periods = 44  # 11 horas = 44 períodos de 15 minutos
        reduced_rest_days = 3  # Número de dias que pode reduzir o repouso para 9 horas (36 períodos)
        reduced_rest_vars = []
        for day in range(num_periods // 96):  # Cada dia tem 96 períodos de 15 minutos
            reduced_rest = solver.BoolVar(f'reduced_rest[{day}]')
            reduced_rest_vars.append(reduced_rest)
            constraint = solver.Add(solver.Sum(X[day * 96:(day + 1) * 96]) <= 96 - (daily_rest_periods - 8 * reduced_rest))
            constraints.append(constraint)
        constraint = solver.Add(solver.Sum(reduced_rest_vars) <= reduced_rest_days)
        constraints.append(constraint)

    # 5. Repouso semanal de 45 horas (180 períodos), pode ser reduzido para 24 horas uma vez a cada duas semanas
    if selected_restrictions["repouso_diario_reduzido"]:    
        reduced_weekly_rest_weeks = 1  # Quantas vezes pode reduzir o repouso para 24 horas a cada duas semanas
        weekly_rest_vars = []
        for week in range(num_periods // (96 * 7)):  # Cada semana tem 96 * 7 períodos
            reduced_rest = solver.BoolVar(f'reduced_weekly_rest[{week}]')
            weekly_rest_vars.append(reduced_rest)
            constraint = solver.Add(solver.Sum(X[week * 96 * 7:(week + 1) * 96 * 7]) >= 180 * (1 - reduced_rest) + 96 * reduced_rest)
            constraints.append(constraint)
        constraint = solver.Add(solver.Sum(weekly_rest_vars) <= reduced_weekly_rest_weeks)
        constraints.append(constraint)

    # 6. Pausa de 45 minutos (3 períodos de 15 minutos) após 4,5 horas de condução (18 períodos)
    if selected_restrictions["pausa_45_minutos"]:    
        for i in range(0, num_periods - 18):  # Verifica cada intervalo de 4,5 horas de condução
            constraint = solver.Add(solver.Sum(X[i:i+18]) <= 18)  # Máximo de 18 períodos de condução (4,5 horas)
            constraints.append(constraint)
            constraint = solver.Add(solver.Sum(X[i+18:i+21]) >= 3)  # Pausa de 3 períodos (45 minutos)
            constraints.append(constraint)

    solver.EnableOutput()
                        
    # Resolver o problema
    status = solver.Solve()

    # Inicializando uma lista para armazenar os resultados
    statisticsResult = []

    # Extrair resultados
    if status == pywraplp.Solver.OPTIMAL:
        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Estado do Modelo: Solução Ótima Encontrada")
        workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
        total_workers = solver.Objective().Value()
    elif status == pywraplp.Solver.FEASIBLE:
        statisticsResult.append(f"Estado do Modelo: Solução Factível Encontrada")
    else:
        statisticsResult.append(f"Estado do Modelo: Solução ótima não encontrada.")
        return 0, [], constraint_matrix, 0, ["Estado do Modelo: Solução ótima não encontrada"]  # Evita retornar None

    # Calcular densidade
    initialDensity = calculate_density(initualconstraintsCoefficients)
    
    # Dentro da sua função solve_shift_schedule
    finalDensity = calculate_density(constraints_coefficients)
    # Calcular densidade
    
    # Adicionando o tipo do modelo aos resultados
    statisticsResult.append(f"Total de trabalhadores necessários: {total_workers}")

    # Adicionando o tipo do modelo aos resultados
    statisticsResult.append(f"Tipo do Modelo: {tipo_modelo(solver)}")

    # Adicionando estatísticas do solver aos resultados
    statisticsResult.append(f"Tempo total de resolução: {solver.wall_time()} ms")
    statisticsResult.append(f"Número total de iterações: {solver.iterations()}")
    statisticsResult.append(f"Número de restrições: {solver.NumConstraints()}")
    statisticsResult.append(f"Número de variáveis: {solver.NumVariables()}")

    return solver, status, total_workers, workers_schedule, constraint_matrix, initualconstraintsCoefficients, constraints_coefficients, initialDensity, finalDensity, statisticsResult

# Interface do Streamlit
st.title("Shift Scheduler")
st.write("""Otimização de agendamento de turnos de motoristas para cobrir a necessidade de cada período de tempo ao longo de um período.""")

# Inicializa a variável default_need vazia
default_need = []
need_input = None
num_periods = None

#Initial
initualconstraintsCoefficients = []

constraints_coefficients = []
# Restrições de cobertura de necessidade em cada período
constraints = []

col1, col2 = st.columns(2)
# Entradas do usuário para horas e períodos na primeira coluna
with col1:
    total_hours = st.number_input("Quantidade total (em horas):", min_value=1, value=1)
    period_minutes = st.number_input("Duração de cada período (em minutos):", min_value=1, value=15)
    variable_type = st.selectbox("Escolha o tipo de variável:", ["Inteira", "Binária", "Contínua"])
    solverParamType = st.selectbox("Use 'GLOP' para problemas de otimização linear contínuos (LP) ou 'SCIP' para problemas de programação inteira mista (MIP)", ["GLOP", "SCIP"])

    
    # Seleção das restrições
    st.write("#### Restrições a serem aplicadas:")
    with st.expander("Restrições", expanded=False):
        selected_restrictions = {}
        for restriction in restrictions:
            checkbox_label = f"{restriction['Descrição']} | {restriction['Fórmula']}"  # Concatenando a descrição e a fórmula
            selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label)
        
# Seleção do tipo de variável na segunda coluna
with col2:
    # #Random
    # default_need = np.random.randint(0, 11, size=(total_hours * 60) // period_minutes).tolist()
    # need_input = st.text_area(f"Demanda de trabalhadores (separado por vírgulas para cada período de {period_minutes} minutos):",', '.join(map(str, default_need)), height=210 )
    # #Exibir a quantidade de elementos em default_need
    # need = [need.strip() for need in need_input.split(',')] 
    # st.write(f"**Número de elementos em 'default_need': {len(need)}**")

    #Manual "1,8,3,4" #
    default_need = "1,8,3,4" # 168 H "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,			8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,	6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,	4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,	8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,	3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"
    #"8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"
    need_input = st.text_area(f"Demanda de trabalhadores (separado por vírgulas para cada período de {period_minutes} minutos):", default_need, height=210 )
    # Exibir a quantidade de elementos em default_need
    need = list(map(int, default_need.split(','))) #[need.strip() for need in need_input.split(',')] 
    st.write(f"**Número de elementos em 'default_need': {len(need)}**")

    # Definindo parâmetros
    num_vars = len(need)  # Número total de variáveis X
    
    num_restricoes = 0
    # Somar apenas as restrições selecionadas
    for restricao in restrictions:
    # Verificar se a restrição está selecionada
        if selected_restrictions.get(restricao["Key"]):  # Verifica se está selecionada e se a Key existe
            num_restricoes += 1  # Incrementa para cada restrição marcada
    # Percorrer cada elemento em default_need e adicioná-lo a rhs_values
    rhs_values = []  # Inicializa rhs_values como uma lista vazia
    for value in need:
        rhs_values.append(value)  # Adiciona cada valor de default_need a rhs_values
    # Gerando e imprimindo a saída do modelo
    model_input = format_lp_output(num_vars, num_restricoes, rhs_values)
    with st.expander("Modelo em LP", expanded=False):
        st.write(f"**Modelo: {model_input}**")
        
# Estrutura de colunas para exibir resultados
col_operations, col_operations2 = st.columns(2)

# Botão para gerar o modelo
num_periods = (total_hours * 60) // period_minutes
generate_model(num_periods, variable_type, selected_restrictions, need)

# Matriz de restrições
num_periods = (total_hours * 60) // period_minutes
constraint_matrix = np.zeros((num_periods, num_periods), dtype=int)

# Preencher a matriz de restrições de acordo com os períodos cobertos
need = list(map(int, need_input.split(',')))
num_periods = len(need)
for i in range(num_periods):
    for j in range(num_periods):
        # Se o período (i - j) está entre os intervalos definidos, marcamos como 1
        if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
            constraint_matrix[i, j] = 1  # Atualiza com cobertura
        else:
            constraint_matrix[i, j] = 0  # Sem cobertura

if num_periods > 0:
    initialDensityMatrix = calculate_density(constraint_matrix)


try:
    need = list(map(int, need_input.split(',')))
    # Estrutura de colunas para exibir resultados
    col_resultsI, col_resultsII = st.columns(2)
    # Exibir a matriz de restrições antes da otimização
    if len(need) != num_periods:
        st.error(f"A entrada deve ter exatamente {num_periods} valores (1 para cada período de {period_minutes} minutos).")
    else:
        with col_resultsI:
            st.subheader("Operações Elementares nas Restrições")
            with st.expander("Operações", expanded=False):
                swap_rows = st.checkbox("Troca de Equações")
                if swap_rows:
                    row1 = st.number_input("Escolha a linha 1 para trocar:", min_value=0, max_value=len(constraint_matrix)-1)
                    row2 = st.number_input("Escolha a linha 2 para trocar:", min_value=1, max_value=len(constraint_matrix)-1)
                    swap_rows = row1, row2
                    solver, status, total_workers, workers_schedule, constraint_matrix, initualconstraintsCoefficients, constraints_coefficients, initialDensity, finalDensity, statisticsResult = solve_shift_schedule(solverParamType, need, variable_type, initualconstraintsCoefficients, constraint_matrix, selected_restrictions, swap_rows=swap_rows, multiply_row=None, add_multiple_rows=None)
                else:
                    solver, status, total_workers, workers_schedule, constraint_matrix, initualconstraintsCoefficients, constraints_coefficients, initialDensity, finalDensity, statisticsResult = solve_shift_schedule(solverParamType, need, variable_type, initualconstraintsCoefficients, constraint_matrix, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None)

                multiply_row_c = st.checkbox("Multiplicação por Constante")
                if multiply_row_c:
                    row = st.number_input("Escolha a linha para multiplicar:", min_value=0, max_value=len(constraint_matrix)-1)
                    constant = st.number_input("Escolha a constante para multiplicar:", value=1)
                    if constant != 0: #and st.button("Aplicar Multiplicação"):
                        multiply_row = (row, constant)
                        solver, status, total_workers, workers_schedule, constraint_matrix, initualconstraintsCoefficients, constraints_coefficients, initialDensity, finalDensity, statisticsResult = solve_shift_schedule(solverParamType, need, variable_type, initualconstraintsCoefficients, constraint_matrix, selected_restrictions, swap_rows=None,multiply_row=multiply_row, add_multiple_rows=None)
                    else:
                        st.warning("A constante de multiplicação não pode ser zero!")
                else:
                    solver, status, total_workers, workers_schedule, constraint_matrix, initualconstraintsCoefficients, constraints_coefficients, initialDensity, finalDensity, statisticsResult = solve_shift_schedule(solverParamType, need, variable_type, initualconstraintsCoefficients, constraint_matrix, selected_restrictions, swap_rows=None,multiply_row=None, add_multiple_rows=None)

                add_multiple_rows_c = st.checkbox("Somar Múltiplo de uma Equação a Outra")
                if add_multiple_rows_c:
                    row1 = st.number_input("Escolha a linha base para somar:", min_value=0, max_value=len(constraint_matrix)-1, key="row1_sum")
                    row2 = st.number_input("Escolha a linha que vai receber o múltiplo:", min_value=0, max_value=len(constraint_matrix)-1, key="row2_sum")
                    multiple = st.number_input("Escolha o múltiplo para somar:", value=0, key="multiple_sum")
                    add_multiple_rows = row1, row2, multiple
                    solver, status, total_workers, workers_schedule, constraint_matrix, initualconstraintsCoefficients, constraints_coefficients, initialDensity, finalDensity, statisticsResult = solve_shift_schedule(solverParamType, need, variable_type, initualconstraintsCoefficients, constraint_matrix, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=add_multiple_rows)
                else:
                    solver, status, total_workers, workers_schedule, constraint_matrix, initualconstraintsCoefficients, constraints_coefficients, initialDensity, finalDensity, statisticsResult = solve_shift_schedule(solverParamType, need, variable_type, initualconstraintsCoefficients, constraint_matrix, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None)
            # Exibir resultados na primeira coluna
        with col_resultsII:
                if total_workers is not None:
                    st.subheader("Resultados")
                    with st.expander("Resultados", expanded=True):
                        # Processar statisticsResult para separar descrições e valores
                        results = {
                            "Descrição": [],
                            "Valor": []
                        }
                        # Preencher o dicionário com os resultados
                        for stat in statisticsResult:
                            # Separar a descrição e o valor usando ':'
                            if ':' in stat:
                                descricao, valor = stat.split(':', 1)  # Divide apenas na primeira ocorrência
                                results["Descrição"].append(descricao.strip())  # Adiciona a descrição sem espaços em branco
                                results["Valor"].append(valor.strip())  # Adiciona o valor sem espaços em branco
                            else:
                                # Caso não haja ':' no stat, adicionar como descrição e valor em branco
                                results["Descrição"].append(stat)
                                results["Valor"].append("")
                        # Criar um DataFrame a partir do dicionário
                        results_df = pd.DataFrame(results)
                        # Exibir o DataFrame como tabela
                        st.table(results_df)
            
                    # Adicionando uma seção colapsável para o escalonamento
                    with st.expander("Escalonamento dos Trabalhadores", expanded=False):
                        st.write("**Escalonamento dos trabalhadores (1 significa que um trabalhador começa neste período):**")
                        st.write(workers_schedule)
                        # Visualização dos dados
                        st.bar_chart(workers_schedule)


    col_resultsIniI, col_resultsIniII, col_resultsIniIII = st.columns(3)
    with col_resultsIniI:
            st.subheader("Matriz de Restrições")
            # Exibir a densidade
            st.write(f"**Densidade da Matriz de Restrições:** {initialDensityMatrix:.4f}")
            with st.expander("Matriz", expanded=True):
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.heatmap(constraint_matrix, cmap="Blues", cbar=True, ax=ax, linewidths=0.01, annot=False, fmt="d")
                plt.title('Matriz de Restrições')
                plt.xlabel('X')
                plt.ylabel('Períodos')
                st.pyplot(fig)
            st.write("A matriz de restrições mostra quais trabalhadores podem cobrir cada período.")
                
    with col_resultsIniII:
        #Exibir a densidade
        st.subheader("Matriz Coeficientes")
        st.write(f"**Densidade Inicial da Matriz Coeficientes:** {initialDensity:.4f}")
        #Exibir a densidade
        with st.expander("Matriz", expanded=True):
            figNew, axNew = plt.subplots(figsize=(14, 8))
            sns.heatmap(initualconstraintsCoefficients, cmap="gray_r", cbar=True, ax=axNew, linewidths=0.01, annot=False, fmt="d")
            plt.title('Matriz de Coeficientes')
            plt.xlabel('X')
            plt.ylabel('Períodos')
            st.pyplot(figNew)
            #initualconstraintsCoefficients[:] = []
    with col_resultsIniIII:
        #Exibir a densidade
        st.subheader("Matriz Coeficientes Pós Operações Elementares")
        st.write(f"**Densidade Final da Matriz Coeficientes:** {finalDensity:.4f}")
        with st.expander("Matriz", expanded=True):
            figNew, axNew = plt.subplots(figsize=(14, 8))
            sns.heatmap(constraints_coefficients, cmap="gray_r", cbar=True, ax=axNew, linewidths=0.01, annot=False, fmt="d")
            plt.title('Matriz de Coeficientes')
            plt.xlabel('X')
            plt.ylabel('Períodos')
            st.pyplot(figNew)
            #constraints_coefficients[:] = []
except Exception as e:
    st.error(f"Ocorreu um erro: {e}")


