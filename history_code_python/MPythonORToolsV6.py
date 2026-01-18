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

# Habilitar a largura total da página
st.set_page_config(layout="wide")

# Função para gerar valores aleatórios e armazená-los no cache
@st.cache_resource
def gerar_valores_aleatorios(total_hours, period_minutes):
    return np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()

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
    {
        "Description": "Split Pause",
        "Formula": r"$\text{Pause} | 15 minutes + 30 minutes$",
        "Detalhes": "The 3 period break can be done as 15 minutes followed by 30 minutes.",
        "Key": "divisao_pausa1530"
    },
        {
        "Description": "Split Pause",
        "Formula": r"$\text{Pause} | 30 minutes + 15 minutes$",
        "Detalhes": "The 3 period break can be done as 30 minutes followed by 15 minutes.",
        "Key": "divisao_pausa3015"
    },
    {
        "Description": "Minimum Daily Rest",
        "Formula": r"$\text{Rest Period} \geq 44p (1p=15minutes |  44p=11h)$",
        "Detalhes": "The driver must rest at least 44 periods (11 hours) every day.",
        "Key": "repouso_diario_minimo"
    },
    {
        "Description": "Weekly Rest",
        "Formula": r"$\text{Rest Period} \geq 180p (1p=15minutes |  180p=45h) $",
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
        "Formula": r"$\sum_{p \in \text{week}} X[p] \leq 224p (1p=15minutes |  224p=56h)$",
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

# Função para relaxar restrições dinamicamente em caso de conflitos
def relax_restrictions(solver, constraints, relaxation_level):
    for constraint in constraints:
        # Ajusta o limite superior ou inferior das restrições para relaxar o problema
        if constraint.Lb() is not None:
            constraint.SetLb(constraint.Lb() - relaxation_level)
        if constraint.Ub() is not None:
            constraint.SetUb(constraint.Ub() + relaxation_level)

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
def solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None, densidadeAceitavel=None, limitWorkers=0,limit_Iteration=0,limit_Level_Relaxation=0, max_slot_Workers=1):
    constraints = []
    msgResult = []

    num_periods = len(need)
    # Criar o solver
    solver = pywraplp.Solver.CreateSolver(solverParamType)  # Usar 'GLOP' para LP ou 'SCIP' para MIP
    
    # Y = {}
    # for d in range(num_periods):
    #     for t in range(limit_Workers):
    #         Y[d, t] = solver.BoolVar(f'Y[{d}, {t}]')
    
   
    
    # Variáveis de decisão
    if variable_type == "Continuous":
        X = [solver.NumVar(0, solver.infinity(), f'X[{d}]') for d in range(num_periods)]
    elif variable_type == "Binary":
        X = [solver.BoolVar(f'X[{d}]') for d in range(num_periods)]
    elif variable_type == "Integer":
        X = [solver.IntVar(0, solver.infinity(), f'X[{d}]') for d in range(num_periods)]
    else:
        raise ValueError("Invalid variable type.")

    if (radio_selection_Object == "Maximizar Atendimento de Demanda"):
        # Função objetivo: minimizar o número total de trabalhadores
        solver.Maximize(solver.Sum(X[d] * need[d] for d in range(num_periods)))
    elif (radio_selection_Object == "Minimizar Número Total de Motorista"):
        # Função objetivo: minimizar o número total de trabalhadores
        solver.Minimize(solver.Sum(X[d] for d in range(num_periods)))
   
    # Restrições de limite diário
    if max_slot_Workers != 0:
        solver.Add(solver.Sum([X[d] for d in range(num_periods)]) <= max_slot_Workers)
    
    if selected_restrictions["cobertura_necessidade"]: 
        for i in range(num_periods):
            constraint_expr = solver.Sum(X[j] for j in range(num_periods) 
                if (0 <= (i - j) % num_periods < 18)  # Janela 0 a 18 (condução inicial)
                
                # Cenario 15/30: Pausa de 15 minutos
                or (19 <= (i - j) % num_periods < 25 and selected_restrictions.get("divisao_pausa1530", False))  # Pausa de 15 minutos
                or (25 <= (i - j) % num_periods < 28 and selected_restrictions.get("divisao_pausa1530", False))
                or (30 <= (i - j) % num_periods < 37 and selected_restrictions.get("divisao_pausa1530", False))
                
                # Cenario 30/15: Pausa de 30 minutos
                or (20 <= (i - j) % num_periods < 21 and selected_restrictions.get("divisao_pausa3015", False))  # Pausa de 30 minutos
                or (21 <= (i - j) % num_periods < 25 and selected_restrictions.get("divisao_pausa3015", False))
                or (26 <= (i - j) % num_periods < 37 and selected_restrictions.get("divisao_pausa3015", False))
                
                # Cenario 45: Pausa de 45 minutos
                or (21 <= (i - j) % num_periods < 37 and selected_restrictions.get("pausa_45_minutos", False))  # Pausa de 45 minutos

                )
            # Adiciona a restrição garantindo que a soma seja maior ou igual à necessidade para o período
            constraint = solver.Add(constraint_expr >= need[i])
            constraints.append(constraint)
    
    # Restrição: Limitar o número total de trabalhadores disponíveis
    if limitWorkers != 0:
        constraint = solver.Add(solver.Sum(X[d] for d in range(num_periods)) <= limitWorkers)
        constraints.append(constraint)

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

    # A multiplicação de uma linha por uma constante é útil se for usada para cancelar ou simplificar valores, mas pode aumentar o fill-in se não for cuidadosamente controlada.
    if multiply_row is not None:
        # Exibir o sistema linear antes da troca de linhas
        row, constant = multiply_row
        if 0 <= row < len(constraints_coefficients):
            if constant != 0:
                # Multiplicação dos coeficientes por uma constante
                constraints_coefficients[row] = [c * constant for c in constraints_coefficients[row]]
                # Atualização da restrição correspondente no solver
                new_expr = solver.Sum(constraints_coefficients[row][j] * X[j] for j in range(len(X)))
                constraints[row] = solver.Add(new_expr >= constraints[row].lb() * constant)
                print(f"Row {row} multiplied by {constant} successfully!")

    #Multiplicacao automatica
    if add_multiple_rows is not None:
        # display_system(constraints_coefficients, need)
        row1, row2, multiple = add_multiple_rows
        if 0 <= row1 < len(constraints_coefficients) and 0 <= row2 < len(constraints_coefficients):
            # Adicionar múltiplo da linha row1 à linha row2
            new_row_values = [
                constraints_coefficients[row2][j] + multiple * constraints_coefficients[row1][j]
                for j in range(len(X))
            ]
        
            # Verificar se a linha resultante tem valores não nulos
            if any(value != 0 for value in new_row_values):
                # Atualizar a linha row2 com os novos valores calculados
                constraints_coefficients[row2] = new_row_values

                # Atualizar o lado direito da equação para a linha row2 no vetor need
                need[row2] = need[row2] + multiple * need[row1]

                # Multiplicar a linha por -1 se houver coeficientes negativos
                if any(value < 0 for value in constraints_coefficients[row2]):
                    constraints_coefficients[row2] = [-value for value in constraints_coefficients[row2]]
                    need[row2] = -need[row2]

                # Atualizar a restrição no solver com os novos valores
                new_expr = solver.Sum(constraints_coefficients[row2][j] * X[j] for j in range(len(X)))
                constraints[row2] = solver.Add(new_expr >= need[row2])

            else:
                st.write("The operation resulted in a null line, which was avoided.")

    # Adicionar as outras restrições conforme o seu código original...
    # 1. Limite diário de condução (9 horas ou 10 horas duas vezes por semana)
    if selected_restrictions["limite_diario"]:    
        for day in range(num_periods // 96):  # 96 períodos de 15 minutos em um dia
            # Cria a expressão para a soma de X[j] nos períodos válidos (condução permitida ou pausas válidas)
            constraint_expr = solver.Sum(
                X[day * 96 + p] for p in range(96)
                if (
                    # Primeira janela: condução inicial (0 a 18 períodos)
                    (0 <= p < 18)
                    # Segunda janela: pausa obrigatória (19 a 20 períodos) Pausa de 15 minutos
                    or (19 <= p < 25 and selected_restrictions.get("divisao_pausa1530", False))
                    or (25 <= p < 28 and selected_restrictions.get("divisao_pausa1530", False))
                    or (30 <= p < 37 and selected_restrictions.get("divisao_pausa1530", False))
                    # Terceira janela: pausa obrigatória (20 a 21 períodos) Pausa de 30 minutos
                    or (20 <= p < 21 and selected_restrictions.get("divisao_pausa3015", False))
                    or (21 <= p < 25 and selected_restrictions.get("divisao_pausa3015", False))
                    or (26 <= p < 37 and selected_restrictions.get("divisao_pausa3015", False))
                    # Quarta janela: pausa obrigatória (21 a 43 períodos) Pausa de 45 minutos
                    or (21 <= p < 37 and selected_restrictions.get("pausa_45_minutos", False))
                )
            )
            
            # Adiciona a restrição de que a soma dos períodos válidos não pode ultrapassar 36
            constraint = solver.Add(constraint_expr <= 36)
            constraints.append(constraint)         

    # 4. Repouso diário mínimo de 11 horas (44 períodos)
    if selected_restrictions["repouso_diario_minimo"]:    
        daily_rest_periods = 44  # 11 horas = 44 períodos de 15 minutos
        reduced_rest_periods = 36  # 9 horas = 36 períodos de 15 minutos
        reduced_rest_days = 3  # Número de dias que pode reduzir o repouso para 9 horas (36 períodos)
        
        reduced_rest_vars = []  # Lista para armazenar as variáveis de repouso reduzido
        
        for day in range(num_periods // 96):  # Para cada dia com 96 períodos de 15 minutos
            reduced_rest = solver.BoolVar(f'reduced_rest[{day}]')  # Variável booleana para indicar repouso reduzido
            reduced_rest_vars.append(reduced_rest)
            
            # Adiciona a restrição para o repouso diário: ou 11 horas ou, caso seja reduzido, 9 horas
            constraint = solver.Add(
                solver.Sum(X[day * 96 + p] for p in range(96))  # Soma os períodos no dia
                <= 96 - (daily_rest_periods - reduced_rest_periods) * reduced_rest  # Ajuste dependendo se o repouso foi reduzido
            )
            constraints.append(constraint)
        
        # Adiciona a restrição que permite no máximo 3 dias com repouso reduzido
        constraint = solver.Add(solver.Sum(reduced_rest_vars) <= reduced_rest_days)
        constraints.append(constraint)
        
    # 5. Repouso semanal de 45 horas (180 períodos), pode ser reduzido para 24 horas uma vez a cada duas semanas
    if selected_restrictions["repouso_diario_reduzido"]:    
        reduced_weekly_rest_weeks = 1  # Quantas vezes pode reduzir o repouso para 24 horas a cada duas semanas
        weekly_rest_vars = []  # Lista para armazenar as variáveis de repouso semanal reduzido

        for week in range(num_periods // (96 * 7)):  # Para cada semana com 96 * 7 períodos de 15 minutos
            reduced_rest = solver.BoolVar(f'reduced_weekly_rest[{week}]')  # Variável booleana para indicar repouso semanal reduzido
            weekly_rest_vars.append(reduced_rest)
            
            # Restrição para o repouso semanal: 
            constraint = solver.Add(
                solver.Sum(X[week * 96 * 7:(week + 1) * 96 * 7]) >= 180 * (1 - reduced_rest) + 96 * reduced_rest
            )
            constraints.append(constraint)

        # Adiciona a restrição que permite no máximo 1 semana com repouso reduzido (24 horas)
        constraint = solver.Add(solver.Sum(weekly_rest_vars) <= reduced_weekly_rest_weeks)
        constraints.append(constraint)

    # Adiciona exceção para 10 horas de condução (40 períodos), no máximo 2 vezes por semana
    if selected_restrictions["limite_semanal"]:    
        extended_hours_days = 2  # Número de dias com 10 horas de condução
        extended_hours_vars = []
        
         # Definir variáveis booleanas para cada dia
        for day in range(num_periods // 96):  # 96 períodos por dia
            extended_hours = solver.BoolVar(f'extended_hours[{day}]')
            extended_hours_vars.append(extended_hours)
            
            # Expressão para somar os períodos válidos (condução ou pausas)
            constraint_expr = solver.Sum(
                X[day * 96 + p] for p in range(96)
                if (
                    # Primeira janela: condução inicial (0 a 18 períodos)
                    (0 <= p < 18)
                    # Segunda janela: pausa obrigatória (19 a 20 períodos) Pausa de 15 minutos
                    or (19 <= p < 25 and selected_restrictions.get("divisao_pausa1530", False))
                    or (25 <= p < 28 and selected_restrictions.get("divisao_pausa1530", False))
                    or (30 <= p < 43 and selected_restrictions.get("divisao_pausa1530", False))
                    # Terceira janela: pausa obrigatória (20 a 21 períodos) Pausa de 30 minutos
                    or (20 <= p < 21 and selected_restrictions.get("divisao_pausa3015", False))
                    or (21 <= p < 25 and selected_restrictions.get("divisao_pausa3015", False))
                    or (26 <= p < 43 and selected_restrictions.get("divisao_pausa3015", False))
                    # Quarta janela: pausa obrigatória (21 a 43 períodos) Pausa de 45 minutos
                    or (21 <= p < 43 and selected_restrictions.get("pausa_45_minutos", False))
                )
            )

            # Restrição de condução no dia (9 ou 10 horas) considerando as pausas
            constraint = solver.Add(constraint_expr <= 40 * extended_hours + 36 * (1 - extended_hours))
            constraints.append(constraint)

        # Restrição para no máximo 2 dias com 10 horas de condução
        constraint = solver.Add(solver.Sum(extended_hours_vars) <= extended_hours_days)
        constraints.append(constraint)

        # Soma total de períodos na semana, com a possibilidade de dias com 10 horas
        weekly_periods = solver.Sum(X[i, j] for i in range(num_periods) for j in range(num_periods))

        # Verifica se a soma de períodos não ultrapassa o limite semanal de 224 períodos
        constraint = solver.Add(weekly_periods <= 224)
        constraints.append(constraint)
    
    if selected_restrictions["descanso_apos_trabalho"]:
        # Iterar sobre os períodos de trabalho
        for i in range(num_periods):  # i representa o período inicial
            for j in range(i + 1, num_periods):  # j representa o período final, buscando a diferença de dias
                diff = abs(i - j)  # A diferença de dias/trabalho consecutivos entre os períodos i e j

                # Condição para descanso após 6 dias de trabalho consecutivos
                if 0 <= diff < 6:  # Após 6 dias de trabalho, é necessário descansar
                    # Adiciona uma restrição no solver para indicar descanso após trabalho
                    constraint = solver.Add(solver.Sum(X[i, j] for i in range(i, j)) <= 0)  # Impede o trabalho sem descanso após 6 dias consecutivos

                    # Esta lógica pode ser personalizada dependendo da forma como os períodos de descanso são distribuídos
                    constraints.append(constraint)

    # 3. Limite quinzenal de condução: 90 horas a cada 14 dias
    if selected_restrictions["limite_quinzenal"]:    
        # Para cada período de 14 dias, com 96 períodos por dia
        for period in range(num_periods // (96 * 14)):  # 96 períodos/dia * 14 dias
            # Expressão para somar os períodos válidos de condução e pausa
            constraint_expr = solver.Sum(
                X[period * 96 * 14 + p] for p in range(96 * 14)
                if (
                    # Primeira janela: condução inicial (0 a 18 períodos)
                    (0 <= p % 96 < 18)
                    # Segunda janela: pausa obrigatória (19 a 20 períodos) Pausa de 15 minutos
                    or (19 <= p % 96 < 25 and selected_restrictions.get("divisao_pausa1530", False))
                    or (25 <= p % 96 < 28 and selected_restrictions.get("divisao_pausa1530", False))
                    or (30 <= p % 96 < 43 and selected_restrictions.get("divisao_pausa1530", False))
                    # Terceira janela: pausa obrigatória (20 a 21 períodos) Pausa de 30 minutos
                    or (20 <= p % 96 < 21 and selected_restrictions.get("divisao_pausa3015", False))
                    or (21 <= p % 96 < 25 and selected_restrictions.get("divisao_pausa3015", False))
                    or (26 <= p % 96 < 43 and selected_restrictions.get("divisao_pausa3015", False))
                    # Quarta janela: pausa obrigatória (21 a 43 períodos) Pausa de 45 minutos
                    or (21 <= p % 96 < 43 and selected_restrictions.get("pausa_45_minutos", False))
                )
            )
            
            # Restrição de condução quinzenal: total de períodos não pode ultrapassar 360
            constraint = solver.Add(constraint_expr <= 360)
            constraints.append(constraint)
        
    # Se "divisao_pausa" também estiver marcado, então aplica a lógica de divisão da pausa
    if selected_restrictions.get("divisao_pausa1530", False):
        # Lógica para dividir a pausa de 45 minutos em 15 e 30 minutos, aplicando a verificação do fracionamento
        for start in range(num_periods - 21):  # Similar ao código anterior: 4,5 horas de condução (18 períodos) + pausa (3 períodos)
            
            # Verifica se é permitido dividir a pausa de acordo com a função de verificação
            if verifica_divisao_pausa(start, start + 1, num_periods):
                # Caso 1: Pausa de 15 minutos seguida de 30 minutos
                first_half_pause = solver.Sum(X[start + 18])  # 15 minutos de pausa (primeiro período de pausa)
                second_half_pause = solver.Sum(X[start + 19])  # 30 minutos de pausa (segundo período de pausa)
                
                # Adiciona a restrição para garantir que a primeira parte da pausa tenha 15 minutos
                constraint_15_30 = solver.Add(first_half_pause >= 1)
                constraints.append(constraint_15_30)
                
                # Adiciona a restrição para garantir que a segunda parte da pausa tenha 30 minutos
                constraint_30_15 = solver.Add(second_half_pause >= 1)
                constraints.append(constraint_30_15)

                # Caso 2: Pausa de 30 minutos seguida de 15 minutos
                second_half_pause = solver.Sum(X[start + 19])  # 30 minutos de pausa (primeiro período de pausa)
                first_half_pause = solver.Sum(X[start + 18])  # 15 minutos de pausa (segundo período de pausa)
                
                # Adiciona a restrição para garantir que a segunda parte da pausa tenha 30 minutos
                constraint_30_15 = solver.Add(second_half_pause >= 1)
                constraints.append(constraint_30_15)

                # Adiciona a restrição para garantir que a primeira parte da pausa tenha 15 minutos
                constraint_15_30_2 = solver.Add(first_half_pause >= 1)
                constraints.append(constraint_15_30_2)

    # # Se "divisao_pausa" também estiver marcado, então aplica a lógica de divisão da pausa
    if selected_restrictions.get("divisao_pausa3015", False):
        # Lógica para dividir a pausa de 45 minutos em 30 minutos + 15 minutos, aplicando a verificação do fracionamento
        for start in range(num_periods - 21):  # 4,5 horas de condução (18 períodos) + pausa (3 períodos)
            
            # Verifica se é permitido dividir a pausa de acordo com a função de verificação
            if verifica_divisao_pausa(start, start + 2, num_periods):
                # Caso 1: Pausa de 30 minutos seguida de 15 minutos
                first_half_pause = solver.Sum(X[start + 18])  # 30 minutos de pausa (primeiro período de pausa)
                second_half_pause = solver.Sum(X[start + 19])  # 15 minutos de pausa (segundo período de pausa)
                
                # Adiciona a restrição para garantir que a primeira parte da pausa tenha 30 minutos
                constraint_30_15 = solver.Add(first_half_pause >= 1)
                constraints.append(constraint_30_15)
                
                # Adiciona a restrição para garantir que a segunda parte da pausa tenha 15 minutos
                constraint_15_30 = solver.Add(second_half_pause >= 1)
                constraints.append(constraint_15_30)

                # Caso 2: Pausa de 15 minutos seguida de 30 minutos
                second_half_pause = solver.Sum(X[start + 19])  # 15 minutos de pausa (primeiro período de pausa)
                first_half_pause = solver.Sum(X[start + 18])  # 30 minutos de pausa (segundo período de pausa)
                
                # Adiciona a restrição para garantir que a segunda parte da pausa tenha 15 minutos
                constraint_15_30_2 = solver.Add(second_half_pause >= 1)
                constraints.append(constraint_15_30_2)

                # Adiciona a restrição para garantir que a primeira parte da pausa tenha 30 minutos
                constraint_30_15_2 = solver.Add(first_half_pause >= 1)
                constraints.append(constraint_30_15_2)
            
    msgResult = None        

    solver.EnableOutput()

    # Dentro da sua função solve_shift_schedule
    finalDensity = calculate_density(constraints_coefficients)
    # Calcular densidade
    initialDensity = calculate_density(initualconstraintsCoefficients)
    
    status = None
    total_workers = 0
    workers_schedule = 0
    
    # Inicializando uma lista para armazenar os resultados
    statisticsResult = []
    iterations_data = []

    
    if finalDensity <= densidadeAceitavel:
        # Resolver o problema
        status = solver.Solve()
        
        # Iterar para resolver conflitos
        max_iterations = limit_Iteration
        relaxation_level = limit_Level_Relaxation  # Relaxa as restrições progressivamente
        iteration = 0
        while status != pywraplp.Solver.OPTIMAL and iteration < max_iterations:
            
            # Capturar dados da iteração
            iteration_data = {
                "iteration": iteration,
                "relaxation_level": relaxation_level,
                "status": get_solver_status_description(status),
                "objective_value": solver.Objective().Value() if status in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE} else 0,
            }
            iterations_data.append(iteration_data)

            # Relaxar restrições relacionadas
            if selected_restrictions["limite_diario"]:
                relax_restrictions(solver, constraints, relaxation_level)
            if selected_restrictions["limite_semanal"]:
                relax_restrictions(solver, constraints, relaxation_level)
            if selected_restrictions["repouso_diario_minimo"]:
                relax_restrictions(solver, constraints, relaxation_level)
    
            # Resolver novamente
            status = solver.Solve()
            iteration += 1
            
            
        if status == pywraplp.Solver.OPTIMAL:
            workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
            total_workers = solver.Objective().Value()
            statisticsResult.append(f"Model State: OPTIMAL")
        elif status == pywraplp.Solver.FEASIBLE:
            statisticsResult.append(f"Model State: FEASIBLE")
        elif status == pywraplp.Solver.INFEASIBLE:
            statisticsResult.append(f"Model State: INFEASIBLE")
        elif status == pywraplp.Solver.UNBOUNDED:
            statisticsResult.append(f"Model State: UNBOUNDED")
        elif status == pywraplp.Solver.ABNORMAL:
            statisticsResult.append(f"Model State: ABNORMAL")
        elif status == pywraplp.Solver.MODEL_INVALID:
            statisticsResult.append(f"Model State: MODEL_INVALID")
        elif status == pywraplp.Solver.NOT_SOLVED:
            statisticsResult.append(f"Model State: NOT_SOLVED")
        else:
            statisticsResult.append(f"Model State: NOT_SOLVED")
            workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
            total_workers = solver.Objective().Value()            

        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Total workers needed: {total_workers}")

        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Model Type: {tipo_modelo(solver)}")

        # Adicionando estatísticas do solver aos resultados
        statisticsResult.append(f"Total Resolution Time: {solver.wall_time()} ms")
        statisticsResult.append(f"Total Number of Iterations: {solver.iterations()}")
        statisticsResult.append(f"Number of Restrictions: {solver.NumConstraints()}")
        statisticsResult.append(f"Number of Variables: {solver.NumVariables()}")
    else:
        statisticsResult.append(f"Model Status: Density not acceptable")
        workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
        total_workers = solver.Objective().Value()

        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Total workers Needed: {total_workers}")

        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Model Type: {tipo_modelo(solver)}")

        # Adicionando estatísticas do solver aos resultados
        statisticsResult.append(f"Total Resolution Time: {solver.wall_time()} ms")
        statisticsResult.append(f"Total Number of Iterations: {solver.iterations()}")
        statisticsResult.append(f"Number of Restrictions: {solver.NumConstraints()}")
        statisticsResult.append(f"Number of Variables: {solver.NumVariables()}")
        
    save_data(constraints_coefficients, 'constraints_coefficients.json')
    
    return solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult, iterations_data

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
def preencher_restricoes(initualconstraintsCoefficients, restrictions, selected_restrictions, num_periods, need):
    # Loop para preencher a matriz de restrições
    for i in range(num_periods):
        for j in range(num_periods):
            # Calcula a diferença cíclica (i - j) com modularidade
            diff = (i - j) % num_periods
            
            # Inicializa como sem cobertura
            initualconstraintsCoefficients[i, j] = 0
            
            # Itera sobre as restrições
            for restriction in restrictions:
                # Verifica se a restrição está selecionada
                if selected_restrictions.get(restriction["Key"], False):
                    
                    # Aplique a lógica para cada restrição
                    if restriction["Key"] == "limite_diario":

                        # Primeira janela: condução inicial (0 a 18 períodos)
                        if 0 <= (i - j) % num_periods < 18:
                            initualconstraintsCoefficients[i, j] = 1  # Condução permitida

                        # Segunda janela: pausa obrigatória (19 a 20 períodos) Pausa de 15 minutos
                        elif 19 <= (i - j) % num_periods < 25 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initualconstraintsCoefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 25 <= (i - j) % num_periods < 28 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initualconstraintsCoefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 30 <= (i - j) % num_periods < 37 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initualconstraintsCoefficients[i, j] = 1  # Pausa fracionada permitida

                        # Terceira janela: pausa obrigatória (19 a 20 períodos) Pausa de 30 minutos
                        elif 20 <= (i - j) % num_periods < 21 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 30 minutos ocupa 1 período
                            initualconstraintsCoefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 21 <= (i - j) % num_periods < 25 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 15 minutos ocupa 1 período
                            initualconstraintsCoefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 26 <= (i - j) % num_periods < 37 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 15 minutos ocupa 1 período
                            initualconstraintsCoefficients[i, j] = 1  # Pausa fracionada permitida

                        # Quarta janela: pausa obrigatória (19 a 20 períodos) Pausa de 30 minutos
                        elif 21 <= (i - j) % num_periods < 37 and selected_restrictions.get("pausa_45_minutos", False):  # Pausa de 30 minutos ocupa 1 período
                            initualconstraintsCoefficients[i, j] = 1  # Pausa fracionada permitida
                            
                        elif 0 <= (i - j) % num_periods < 44 and selected_restrictions.get("repouso_diario_minimo", False):
                            # Exemplo de condição para repouso diário mínimo
                            #if 0 <= diff < 44:  # Repouso diário de 11h
                            initualconstraintsCoefficients[i, j] = 1    
                            
                        elif 0 <= (i - j) % num_periods < 36 and selected_restrictions.get("repouso_diario_reduzido", False): #restriction["Key"] == "repouso_diario_reduzido":
                            # Exemplo de condição para repouso diário reduzido
                                # if 0 <= diff < 36:  # Repouso de 9h
                            initualconstraintsCoefficients[i, j] = 1                                                    
  
                        # Fora dos intervalos permitidos
                        else:
                            initualconstraintsCoefficients[i, j] = 0  # Fora do intervalo permitido

                    elif restriction["Key"] == "repouso_semanal":
                        # Exemplo de condição para repouso semanal
                        if 0 <= diff < 180:  # Repouso semanal de 45h
                            initualconstraintsCoefficients[i, j] = 1

                    elif restriction["Key"] == "repouso_quinzenal":
                        # Exemplo de condição para repouso quinzenal
                        if 0 <= diff < 96:  # Repouso quinzenal de 24h
                            initualconstraintsCoefficients[i, j] = 1

                    elif restriction["Key"] == "descanso_apos_trabalho":
                        # Exemplo de condição para descanso após dias de trabalho
                        if 0 <= diff < 6:  # Após 6 dias de trabalho, descanso
                            initualconstraintsCoefficients[i, j] = 1

                    elif restriction["Key"] == "limite_semanal":
                        # Limite semanal de condução (224 períodos)
                        # Verifica a soma de períodos dentro de uma semana
                        week_start = (i // 7) * 7
                        week_end = week_start + 7
                        weekly_periods = np.sum(initualconstraintsCoefficients[week_start:week_end, :])
                        if weekly_periods <= 224:
                            initualconstraintsCoefficients[i, j] = 1

                    elif restriction["Key"] == "limite_quinzenal":
                        # Limite quinzenal de condução (360 períodos)
                        # Verifica a soma de períodos dentro de uma quinzena
                        fortnight_start = (i // 14) * 14
                        fortnight_end = fortnight_start + 14
                        fortnight_periods = np.sum(initualconstraintsCoefficients[fortnight_start:fortnight_end, :])
                        if fortnight_periods <= 360:
                            initualconstraintsCoefficients[i, j] = 1

                    elif restriction["Key"] == "cobertura_necessidade":
                        # Cobertura de necessidade (verifica a necessidade de trabalhadores)
                        if np.sum(initualconstraintsCoefficients[:, i]) >= need[i]:
                            initualconstraintsCoefficients[i, j] = 1

                            
    return initualconstraintsCoefficients

# Interface do Streamlit
st.title("Shift Scheduler")

# Inicializa a variável default_need vazia
default_need = []
need_input = None
num_periods = None

initualconstraintsCoefficients = []
# Carrega os dados do arquivo, se existirem
initualconstraintsCoefficients = load_data('initualconstraintsCoefficients.json')

# Criando o formulário
with st.form("paramModel"):
    with st.expander("", expanded=True):    
        # Colunas do formulário
        col1, col2, col3, col4 = st.columns(4)
        
        # Entradas do usuário para horas e períodos na primeira coluna
        with col1:
            st.write("Global")
            total_hours = st.number_input("Hour", min_value=1, value=1)
            period_minutes = st.number_input("Slot", min_value=1, value=15)
            limit_Workers = st.number_input("Drivers (0, no limit)", min_value=0)
            max_slot_Workers = st.number_input("Slots (0, no limit)", min_value=0)
            
           
        
        with col2:
            st.write("Algorithm")
            variable_type = st.selectbox("Variable", ["Integer", "Binary", "Continuous"])
            solverParamType = st.selectbox("GLOP-LP | SCIP-MIP", ["SCIP", "GLOP"])
            acceptable_percentage = st.number_input("Acceptable Density", min_value=0.01)
            
        with col3:
            st.write("Iterations|Relaxation")
            limit_Iteration = st.number_input("Limit Iterations", min_value=0, value=0)
            limit_Level_Relaxation = st.number_input("Relaxation", min_value=0, value=0)
        with col4:
            fixar_valores = st.checkbox("Set Values", value=True)
            # Condicional para definir `default_need` com valores fixos ou novos aleatórios
            if fixar_valores:
                default_need = gerar_valores_aleatorios(total_hours, period_minutes)
            else:
                default_need = np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()
            
            # default_need = np.random.randint(0, 11, size=(total_hours * 60) // period_minutes).tolist()
            need_input = st.text_area(f"Slot Demand",', '.join(map(str, default_need)), height=210 )
            #Exibir a quantidade de elementos em default_need
            need = [need.strip() for need in need_input.split(',')] 
            st.write(f"Total Demand {len(need)}")
    
    # Seleção das restrições
    st.subheader("Restrictions")
    with st.expander("Global", expanded=True):
        
        selected_restrictions = {}
        col1, col2, col3 = st.columns([2, 4, 5])

        # Radiobuttons na primeira coluna
        with col1:
            
            st.write("Função Objetivo")
            radio_selection_Object = st.radio(
                "Selecione o objetivo", 
                options=["Maximizar Atendimento de Demanda", "Minimizar Número Total de Motorista"],
                index=0, 
                key="funcao_Objetivo"
            )
            
            st.write("Break Options")
            radio_selection = st.radio(
                "Select pause", 
                options=["None", "45 minutes", "15+30 split", "30+15 split"],
                index=0,  # Inicialmente nenhuma pausa selecionada
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
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"], value=default_checked, disabled=False)
                elif restriction["Key"] == "limite_diario":
                    default_checked = restriction["Key"] == "limite_diario"
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"], value=default_checked, disabled=False)
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

    # Botão de submit do formulário
    submit_button = st.form_submit_button("Apply Template")               
        
    if submit_button:
        
        # Processamento dos dados de demanda
        need = [need.strip() for need in need_input.split(',')]
        st.write(f"Total Demand {len(need)}")
        
        # Calculando o número total de variáveis e restrições
        num_vars = len(need)
        num_restricoes = sum(1 for restricao in restrictions if selected_restrictions.get(restricao["Key"]))

        # Preenchendo os valores RHS com a demanda
        rhs_values = [int(value) for value in need]
        
        model_input = format_lp_output(num_vars, num_restricoes, rhs_values)
        with st.expander("PL model", expanded=False):
            st.write(f"**Model: {model_input}**")

# Estrutura de colunas para exibir resultados
col_operations, col_operations2 = st.columns(2)
num_periods = (total_hours * 60) // period_minutes
# Matriz de restrições
num_periods = (total_hours * 60) // period_minutes
# Preencher a matriz de restrições de acordo com os períodos cobertos
need = list(map(int, need_input.split(',')))
num_periods = len(need)
num_dias = total_hours // 24

initualconstraintsCoefficients = np.zeros((num_periods, num_periods), dtype=int)
initualconstraintsCoefficients = preencher_restricoes(initualconstraintsCoefficients, restrictions, selected_restrictions, num_periods, need)

# Se não houver dados no arquivo, inicializa com uma matriz padrão
constraints_coefficients = load_data('constraints_coefficients.json') 
if constraints_coefficients is None or constraints_coefficients.size == 0:
    constraints_coefficients = initualconstraintsCoefficients
    save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo

st.subheader("Cache")
with st.expander("Parameters", expanded=True):
    atualizarFicheiro = st.checkbox("Update File", value=False)
    if atualizarFicheiro:
        constraints_coefficients = initualconstraintsCoefficients
        save_data(initualconstraintsCoefficients, 'initualconstraintsCoefficients.json')
        save_data(constraints_coefficients, 'constraints_coefficients.json')

    if num_periods > 0:
        initialDensityMatrix = calculate_density(initualconstraintsCoefficients)
        initialDensityMatrix = initialDensityMatrix 
        st.write(f"Initial Density {initialDensityMatrix:.2f}%")

st.subheader("Elementary Operations")
# Loop para gerar as checkboxes com base nas operações
selected_operations = {}
for elementalOperation in elementalOperations:
    checkbox_label = f"{elementalOperation['Description']} | {elementalOperation['Formula']}"
    selected_operations[elementalOperation["Key"]] = st.checkbox(checkbox_label, key=elementalOperation["Key"], value=False)

# Inicializar variáveis para as operações
swap_rows = None
multiply_row = None
add_multiple_rows = None


msgResult = []
iterations_dataResult = []
try:
    need = list(map(int, need_input.split(',')))
    # Estrutura de colunas para exibir resultados
    col_resultsItIOpI,col_resultsItIOpII, col_resultsItIOpIII = st.columns(3)
    col_resultsI = st.columns(1)[0]
    col_resultsItI = st.columns(1)[0]


    # Exibir a matriz de restrições antes da otimização
    if len(need) != num_periods:
        st.error(f"The input must have exactly {num_periods} values ​​(1 for each period of {period_minutes} minutes).")
    else:
        with col_resultsI:
            # st.subheader("Operações Elementares")
            with st.expander("Operations", expanded=True):
                
                swap_rows_c = selected_operations["troca_equacoes"] #st.checkbox("Troca de Equações")
                multiply_row_c = selected_operations["multiplicacao_por_constante"] #st.checkbox("Multiplicação por Constante")
                add_multiple_rows_c = selected_operations["soma_multiplo_equacao"] #st.checkbox("Somar Múltiplo de uma Equação a Outra")
                add_multiple_rows_c_auto = selected_operations["soma_multiplo_equacao_automatica"] #st.checkbox("Somar Múltiplo de uma Equação a Outra - Automático")
                
                if (not swap_rows_c 
                    and not multiply_row_c 
                        and not add_multiple_rows_c
                        and not add_multiple_rows_c_auto):
                        constraints_coefficients = load_data('constraints_coefficients.json') 
                        solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult, iterations_dataResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None,densidadeAceitavel=acceptable_percentage,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation, max_slot_Workers=max_slot_Workers )
                else:
                    if swap_rows_c:
                        with col_resultsItIOpI:
                            constraints_coefficients = load_data('constraints_coefficients.json') 
                            row1 = st.number_input("Choose line 1 to exchange", min_value=0, max_value=len(constraints_coefficients)-1)
                            row2 = st.number_input("Choose line 2 to exchange:", min_value=1, max_value=len(constraints_coefficients)-1)
                            swap_rows = row1, row2
                            constraints_coefficients = load_data('constraints_coefficients.json')
                            solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult, iterations_dataResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=swap_rows, multiply_row=None, add_multiple_rows=None,densidadeAceitavel=acceptable_percentage,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation, max_slot_Workers=max_slot_Workers)
                    if multiply_row_c:
                        with col_resultsItIOpII:
                            constraints_coefficients = load_data('constraints_coefficients.json') 
                            row = st.number_input("Choose the line to multiply", min_value=0, max_value=len(constraints_coefficients)-1)
                            constant = st.number_input("Choose the constant to multiply", value=1)
                            if constant != 0: #and st.button("Aplicar Multiplicação"):
                                multiply_row = (row, constant)
                                constraints_coefficients = load_data('constraints_coefficients.json')
                                solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult, iterations_dataResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None,multiply_row=multiply_row, add_multiple_rows=None,densidadeAceitavel=acceptable_percentage,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation, max_slot_Workers=max_slot_Workers)
                            else:
                                st.warning("The multiplication constant cannot be zero!")
                    if add_multiple_rows_c:
                        with col_resultsItIOpIII:
                            constraints_coefficients = load_data('constraints_coefficients.json')
                            row1 = st.number_input("Choose the baseline to sum", min_value=0, max_value=len(constraints_coefficients)-1, key="row1_sum")
                            row2 = st.number_input("Choose the line that will receive the multiple", min_value=0, max_value=len(constraints_coefficients)-1, key="row2_sum")
                            multiple = st.number_input("Choose the multiple to add", value=0, key="multiple_sum")
                            add_multiple_rows = row1, row2, multiple
                            solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult, iterations_dataResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=add_multiple_rows,densidadeAceitavel=acceptable_percentage,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation, max_slot_Workers=max_slot_Workers)
                    if add_multiple_rows_c_auto:
                        constraints_coefficients = load_data('constraints_coefficients.json')
                        #Percorrer a matriz com o multiplicador -1, todas as linhas considerando o row2 sempre um a menor que o row1 e submetendo ao modelo.
                        for idx, row in enumerate(constraints_coefficients):
                            row1 = idx+1
                            row2 = idx
                            multiple = -1
                            add_multiple_rows = row1, row2, multiple
                            solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult, iterations_dataResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=add_multiple_rows,densidadeAceitavel=acceptable_percentage,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation, max_slot_Workers=max_slot_Workers)
                            # Dentro da sua função solve_shift_schedule
                            finalDensity = calculate_density(constraints_coefficients)
                            if finalDensity <= acceptable_percentage:
                                st.write(f"Final density {finalDensity} has reached acceptable limit ({acceptable_percentage}). Exiting loop.")
                                break
            # Exibir resultados na primeira coluna
            with col_resultsItI:
                    if msgResult is None:
                        if total_workers is not None:
                                st.subheader("Results")
                                with st.expander("Results", expanded=True):
                                    # Processar statisticsResult para separar descrições e valores
                                    results = {
                                        "Description": [],
                                        "Value": []
                                    }
                                    # Preencher o dicionário com os resultados
                                    for stat in statisticsResult:
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
                        
                                with st.expander("Customers Demand", expanded=True):
                                    # Converter a entrada de texto para uma lista de números
                                    try:
                                        demanda = list(map(int, need_input.split(',')))
                                    except ValueError:
                                        st.error("Por favor, insira os valores da demanda separados por vírgula e espaço.")
                                        demanda = []
                            
                                    #Gerar um DataFrame com os dados
                                    slots = list(range(1, len(demanda) + 1))  # Slots de 1 a 96
                                    df_demanda = pd.DataFrame({
                                        "Slot": slots,
                                        "Demanda": demanda
                                    })

                                    # Exibir o gráfico de barras utilizando st.bar_chart
                                    st.bar_chart(df_demanda.set_index("Slot"))

                        
                                # Adicionando uma seção colapsável para o escalonamento
                                with st.expander("Driver Scheduling", expanded=True):
                                    # Verificar se workers_schedule possui valores maiores que 0
                                    if workers_schedule and any(value > 0 for value in workers_schedule):
                                        # Converter workers_schedule para um DataFrame para visualização
                                        schedule_df = pd.DataFrame({
                                            "Period": list(range(1, len(workers_schedule) + 1)),
                                            "Driver": workers_schedule
                                        })
                                        
                                        st.bar_chart(schedule_df.set_index("Period"))
                                    else:
                                        st.warning("Driver schedule is empty or does not contain valid values.")
                                        
                                                                                                # Gerar DataFrame para os dados
                                with st.expander("Comparação entre Demanda de Clientes e Escalonamento de Motoristas", expanded=True):
                                    # Gerar DataFrame para os dados
                                    slots = list(range(1, len(demanda) + 1))  # Slots de 1 a 96
                                    df_comparacao = pd.DataFrame({
                                        "Slot": slots,
                                        "Demanda": demanda,
                                        "Motoristas": workers_schedule
                                    })

                                    # Calcular Taxa de Cobertura
                                    df_comparacao['Taxa de Cobertura'] = df_comparacao['Motoristas'] / df_comparacao['Demanda']

                                    # Calcular Índice de Sobrecarga (excesso de motoristas)
                                    df_comparacao['Índice de Sobrecarga'] = (df_comparacao['Motoristas'] - df_comparacao['Demanda']) / df_comparacao['Demanda']
                                    df_comparacao['Índice de Sobrecarga'] = df_comparacao['Índice de Sobrecarga'].apply(lambda x: x if x > 0 else 0)

                                    # Calcular Desvio Padrão da Eficiência (taxa de cobertura)
                                    desvio_padrao = np.std(df_comparacao['Taxa de Cobertura'])

                                    # Calcular Índice de Subutilização
                                    df_comparacao['Índice de Subutilização'] = (df_comparacao['Demanda'] - df_comparacao['Motoristas']) / df_comparacao['Motoristas']
                                    df_comparacao['Índice de Subutilização'] = df_comparacao['Índice de Subutilização'].apply(lambda x: x if x > 0 else 0)

                                    # Exibir as métricas
                                    st.title("Eficiência da Alocação de Motoristas")
                                    st.write(f"**Desvio Padrão da Taxa de Cobertura:** {desvio_padrao:.2f}")

                                    # Determinar a cor com base no desvio padrão
                                    if desvio_padrao < 0.2:
                                        desvio_cor = "green"
                                        status = "Bom"
                                    elif desvio_padrao < 0.5:
                                        desvio_cor = "orange"
                                        status = "Razoável"
                                    else:
                                        desvio_cor = "red"
                                        status = "Ruim"

                                    # Exibir o desvio padrão da taxa de cobertura com cor
                                    # st.markdown(f"**Desvio Padrão da Taxa de Cobertura:** <span style='color:{desvio_cor};'>{desvio_padrao:.2f} ({status})</span>", unsafe_allow_html=True)

                                    # Exibir o desvio padrão da taxa de cobertura com cor e formatação em negrito
                                    st.markdown(f"<h3 style='color:{desvio_cor}; font-weight: bold;'>Desvio Padrão da Taxa de Cobertura: {desvio_padrao:.2f} ({status})</h3>", unsafe_allow_html=True)



                                    # Exibir o DataFrame com as métricas calculadas
                                    st.write("Tabela com as métricas de eficiência:")
                                    st.dataframe(df_comparacao)
                                    
                                    # Calcular e exibir os totais para cada indicador
                                    total_demanda = df_comparacao['Demanda'].sum()
                                    total_motoristas = df_comparacao['Motoristas'].sum()
                                    total_taxa_cobertura = df_comparacao['Taxa de Cobertura'].mean()
                                    total_sobrecarga = df_comparacao['Índice de Sobrecarga'].sum()
                                    total_subutilizacao = df_comparacao['Índice de Subutilização'].sum()

                                    # Exibir gráfico com barras comparativas
                                    st.bar_chart(df_comparacao.set_index("Slot")[["Demanda", "Motoristas"]])

                                    # Exibir o DataFrame com as métricas calculadas
                                    st.write("Tabela com as métricas de eficiência:")
                                    st.dataframe(df_comparacao)

                                    # Gráfico da Taxa de Cobertura
                                    st.subheader(f"Taxa de Cobertura por Slot (Total: {total_taxa_cobertura:.2f})")
                                    st.bar_chart(df_comparacao.set_index("Slot")['Taxa de Cobertura'])

                                    # Gráfico do Índice de Sobrecarga
                                    st.subheader(f"Índice de Sobrecarga por Slot (Total: {total_sobrecarga:.2f})")
                                    st.bar_chart(df_comparacao.set_index("Slot")['Índice de Sobrecarga'])

                                    # Gráfico do Índice de Subutilização
                                    st.subheader(f"Índice de Subutilização por Slot (Total: {total_subutilizacao:.2f})")
                                    st.bar_chart(df_comparacao.set_index("Slot")['Índice de Subutilização'])

                                    # Gráfico Comparativo entre Demanda e Motoristas
                                    st.subheader(f"Comparação entre Demanda e Motoristas (Demanda Total: {total_demanda}, Motoristas Total: {total_motoristas})")
                                    st.bar_chart(df_comparacao.set_index("Slot")[["Demanda", "Motoristas"]])

                                    # Exibir o gráfico de barras com st.bar_chart
                                    # st.title("Comparação entre Demanda de Clientes e Escalonamento de Motoristas")

                                    # Exibir o gráfico com barras lado a lado
                                    # st.bar_chart(df_comparacao.set_index("Slot"))        
                                        
                                        
                                
        
        col_resultsIniI, col_resultsIniII = st.columns(2)
        with col_resultsIniI:
            if msgResult is None:    
                if initialDensityMatrix is not None:
                    st.subheader("Initial Constraint Matrix")
                    # Exibir a densidade
                    st.write(f"Density: {initialDensityMatrix:.4f}")
                    with st.expander("Matrix", expanded=True):
                        fig, ax = plt.subplots(figsize=(14, 8))
                        sns.heatmap(initualconstraintsCoefficients, cmap="Blues", cbar=False, annot=False, fmt="d", annot_kws={"size": 7})
                        plt.title('Constraint Matrix')
                        plt.xlabel('X')
                        plt.ylabel('Period')
                        st.pyplot(fig)
                    # st.write("The constraint matrix shows how many drivers can cover each period.")
                
                 # Converter dados para DataFrame
            if iterations_dataResult != []:
                st.subheader("Convergence Progress")
                df_iterationsResult = pd.DataFrame(iterations_dataResult)
                # # Gráfico de convergência do objetivo
                fig, ax = plt.subplots(figsize=(14, 8))
                df_iterationsResult.plot(x="iteration", y="objective_value", ax=ax, marker="o", label="Goal Value")
                ax.set_title("Convergence of Result")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Goal Value")
                ax.grid(True)
                ax.legend()
                # Exibir o gráfico no Streamlit
                st.pyplot(fig)    
                    
                    
        with col_resultsIniII:
            if msgResult is None:
                #Exibir a densidade
                st.subheader("Final Constraints Matrix")
                st.write(f"Final Density Matrix Constraints: {finalDensity:.4f}")
                with st.expander("Matriz", expanded=True):
                    figNew, axNew = plt.subplots(figsize=(14, 8))
                    constraints_coefficients = load_data('constraints_coefficients.json')
                    sns.heatmap(constraints_coefficients, cmap="Oranges", cbar=False, annot=False, fmt="d", annot_kws={"size": 6})
                    plt.title('Constraints Matrix')
                    plt.xlabel('X')
                    plt.ylabel('Period')
                    st.pyplot(figNew)
                    
            if iterations_dataResult != []:
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
        
        # col_resultsInterationI, col_resultsInterationII, col_resultsInterationIII, col_resultsInterationIIV = st.columns(4)
        # with col_resultsInterationI:
            # # Converter dados para DataFrame
            # if iterations_dataResult != []:
            #     df_iterationsResult = pd.DataFrame(iterations_dataResult)
            #     # # Gráfico de convergência do objetivo
            #     fig, ax = plt.subplots(figsize=(14, 8))
            #     df_iterationsResult.plot(x="iteration", y="objective_value", ax=ax, marker="o", label="Goal Value")
            #     ax.set_title("Convergence of Result", fontsize=6)
            #     ax.set_xlabel("Iteration", fontsize=4)
            #     ax.set_ylabel("Goal Value", fontsize=4)
            #     ax.grid(True)
            #     ax.legend()
            #     # Exibir o gráfico no Streamlit
            #     st.pyplot(fig)
        # with col_resultsInterationII:
            # if iterations_dataResult != []:
            #     fig_relax, ax_relax = plt.subplots(figsize=(14, 8))
            #     df_iterationsResult.plot(x="iteration", y="relaxation_level", ax=ax_relax, marker="x", color="red", label="Relaxation Level")
            #     ax_relax.set_title("Relaxation Progress", fontsize=6)
            #     ax_relax.set_xlabel("Iteration", fontsize=4)
            #     ax_relax.set_ylabel("Relaxation Level", fontsize=4)
            #     ax_relax.grid(True)
            #     ax_relax.legend()

            #     # Exibir o gráfico no Streamlit
            #     st.pyplot(fig_relax)
        if iterations_dataResult != []:            
            st.table(df_iterationsResult)
        if msgResult != None:
            # Criando o DataFrame a partir da lista msgResult
            resultsMsgResult = {
                "Description": [],
                "Value": []
            }
            # Preencher o dicionário com os resultados
            for stat in msgResult:
                # Separar a descrição e o valor usando ':'
                if ':' in stat:
                    descricao, valor = stat.split(':', 1)  # Divide apenas na primeira ocorrência
                    resultsMsgResult["Description"].append(descricao.strip())  # Adiciona a descrição sem espaços em branco
                    resultsMsgResult["Value"].append(valor.strip())  # Adiciona o valor sem espaços em branco
                else:
                    # Caso não haja ':' no stat, adicionar como descrição e valor em branco
                    resultsMsgResult["Description"].append(stat)
                    resultsMsgResult["Value"].append("")
            # Criar um DataFrame a partir do dicionário
            dfmsgResult = pd.DataFrame(resultsMsgResult)
            # Exibir o DataFrame como tabela
            st.table(dfmsgResult)

except Exception as e:
    st.error(f"Ocorreu um erro: {e}")