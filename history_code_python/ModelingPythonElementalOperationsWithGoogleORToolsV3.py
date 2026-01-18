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
from math import ceil
import plotly.express as px
import plotly.graph_objects as go

# Habilitar a largura total da página
st.set_page_config(layout="wide")

# Função para gerar valores aleatórios e armazená-los no cache
@st.cache_resource
def gerar_valores_aleatorios(total_hours, period_minutes):
    return np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()
  
# Definindo as restrições
restrictions = [
    {
        "Descrição": "Cobertura de necessidade",
        "Fórmula": r"$\sum_{j \in \text{janelas válidas}(i)} X[j] \geq \text{need}[i]$",
        "Detalhes": "O número de trabalhadores alocados em períodos que atendem as janelas válidas deve ser suficiente para satisfazer a necessidade mínima do período \(i\).",
        "Key": "cobertura_necessidade"
    },
    {
        "Descrição": "Limite diário de condução",
        "Fórmula": r"$\sum_{p \in \text{dia}} X[p] \leq 36p (1p=15mim, logo  36p=9h)$",
        "Detalhes": "Pode ser 40 períodos (10 horas) duas vezes por semana.",
        "Key": "limite_diario"
    },
    {
        "Descrição": "Limite semanal de condução",
        "Fórmula": r"$\sum_{p \in \text{semana}} X[p] \leq 224p (1p=15mim, logo  224p=56h)$",
        "Detalhes": "Total de períodos de trabalho durante uma semana não deve ultrapassar 224 períodos.",
        "Key": "limite_semanal"
    },
    {
        "Descrição": "Limite quinzenal de condução",
        "Fórmula": r"$\sum_{p \in \text{quinzena}} X[p] \leq 360p (1p=15mim, logo  360p=90h)$",
        "Detalhes": "Total de períodos de trabalho em duas semanas não deve ultrapassar 360 períodos.",
        "Key": "limite_quinzenal"
    },
    {
        "Descrição": "Repouso diário mínimo",
        "Fórmula": r"$\text{Repouso} \geq 44p (1p=15mim, logo  44p=11h)$",
        "Detalhes": "O motorista deve descansar pelo menos 44 períodos (11 horas) todos os dias.",
        "Key": "repouso_diario_minimo"
    },
    {
        "Descrição": "Repouso diário reduzido",
        "Fórmula": r"$\text{Repouso} \geq 36p \text{ (1p=15mim, logo  36p=9h | máx. 3 vezes em 14 dias)}$",
        "Detalhes": "O repouso pode ser reduzido para 36 períodos (9 horas), mas não mais do que 3 vezes em 14 dias.",
        "Key": "repouso_diario_reduzido"
    },
    {
        "Descrição": "Repouso semanal",
        "Fórmula": r"$\text{Repouso semanal} \geq 180p (1p=15mim, logo  180p=45h) $",
        "Detalhes": "O motorista deve ter um período de repouso de 180 períodos (45 horas) toda semana.",
        "Key": "repouso_semanal"
    },
    {
        "Descrição": "Repouso quinzenal",
        "Fórmula": r"$\text{Repouso quinzenal} \geq 96p (1p=15mim, logo  96p=24h)$",
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
        "Fórmula": r"$\text{Pausa} \geq 3p (1p=15mim, logo 3p=45min)$",
        "Detalhes": "Após no máximo 18 períodos (4,5 horas), deve haver uma pausa de 3 períodos (45 minutos).",
        "Key": "pausa_45_minutos"
    },
    {
        "Descrição": "Divisão da pausa",
        "Fórmula": r"$\text{Pausa} = \text{1 período de 15 minutos} + \text{1 período de 30 minutos}$",
        "Detalhes": "A pausa de 3p períodos pode ser feita como 15 minutos seguidos de 30 minutos.",
        "Key": "divisao_pausa"
    }
]

# Função para salvar os dados no arquivo
def save_data(data, FILENAME):
    with open(FILENAME, 'w') as f:
        json.dump(data.tolist(), f)  # Converte a matriz NumPy para lista
        
#     # Função para carregar o estado da matriz do arquivo JSON, se ele existir
# Função para carregar os dados do arquivo
def load_data(FILENAME):
    if os.path.exists(FILENAME):
        try:
            with open(FILENAME, 'r') as f:
                data = json.load(f)
                return np.array(data)  # Converte a lista de volta para uma matriz NumPy
        except json.JSONDecodeError:
            st.error("O arquivo JSON está corrompido ou inválido. Um novo arquivo será criado.")
            # Aqui você pode optar por apagar o arquivo corrompido ou criar um novo
            # os.remove(FILENAME)  # Descomente se quiser excluir o arquivo
            return None  # Retorna None se ocorrer um erro
    return None  # Retorna None se o arquivo não existir


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

    return output

# Função para exibir o modelo de otimização
def generate_model(num_periods, variable_type, selected_restrictions, need):
 
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

# # Função para aplicar operações elementares
# def apply_elementary_operations(constraints_coefficients):
#     #with st.expander("Matriz de restrições pós operações:", expanded=False):
#     st.subheader("Operações Elementares nas Restrições")
#     with st.expander("Escalonamento dos Trabalhadores", expanded=False):
#         swap_rows = st.checkbox("Troca de Equações")
#         multiply_row = st.checkbox("Multiplicação por Constante")
#         add_multiple_rows = st.checkbox("Somar Múltiplo de uma Equação a Outra")

#         if swap_rows:
#                 row1 = st.number_input("Escolha a linha 1 para trocar:", min_value=0, max_value=len(constraints_coefficients)-1)
#                 row2 = st.number_input("Escolha a linha 2 para trocar:", min_value=0, max_value=len(constraints_coefficients)-1)
#                 if st.button("Aplicar Troca"):
#                     constraints_coefficients[[row1, row2]] = constraints_coefficients[[row2, row1]]
#                     st.success(f"Linhas {row1} e {row2} trocadas com sucesso!")


#         if multiply_row:
#             row = st.number_input("Escolha a linha para multiplicar:", min_value=0, max_value=len(constraints_coefficients)-1)
#             constant = st.number_input("Escolha a constante para multiplicar:", value=1.0)
#             if constant != 0 and st.button("Aplicar Multiplicação"):
#                 constraints_coefficients = constraints_coefficients.astype(np.float64)
#                 constraints_coefficients[row] *= constant
#                 st.success(f"Linha {row} multiplicada por {constant} com sucesso!")
#             else:
#                 st.warning("A constante de multiplicação não pode ser zero!")

#         if add_multiple_rows:
#             row1 = st.number_input("Escolha a linha base para somar:", min_value=0, max_value=len(constraints_coefficients)-1, key="row1_sum")
#             row2 = st.number_input("Escolha a linha que vai receber o múltiplo:", min_value=0, max_value=len(constraints_coefficients)-1, key="row2_sum")
#             multiple = st.number_input("Escolha o múltiplo para somar:", value=1.0, key="multiple_sum")
#             if st.button("Aplicar Soma"):
#                 constraints_coefficients = constraints_coefficients.astype(np.float64)
#                 constraints_coefficients[row2] += multiple * constraints_coefficients[row1]
#                 st.success(f"Múltiplo da linha {row1} somado à linha {row2} com sucesso!")
        
#         st.write(constraints_coefficients)

#     return constraints_coefficients

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

        
# Função de otimização (baseada no código anterior)
def solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None, densidadeAceitavel=None, limitWorkers=0):
    constraints = []
    msgResult = []
    
    num_periods = len(need)
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

    if selected_restrictions["cobertura_necessidade"]: 
        for i in range(num_periods):
            constraint = solver.Add(solver.Sum(X[j] for j in range(num_periods) 
                                if (0 <= (i - j) % num_periods < 18)  
                                or (19 <= (i - j) % num_periods < 23)  
                                or (25 <= (i - j) % num_periods < 43) ) >= need[i])
            constraints.append(constraint)
    
    
    # Restrição: Limitar o número total de trabalhadores disponíveis
    if limitWorkers != 0:
        constraint = solver.Add(solver.Sum(X[d] for d in range(num_periods)) <= limitWorkers)
        constraints.append(constraint)

    # Aplicar operações elementares, se selecionadas
    #A troca de linhas não reduz diretamente o fill-in, mas pode ajudar na preparação de operações de eliminação.
    # Exemplo de troca de linhas na matriz de coeficientes
    if swap_rows is not None:
        # Exibir o sistema linear antes da troca de linhas
        st.write("Sistema Linear antes da troca de linhas:")
        display_system(initualconstraintsCoefficients,need)
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
        display_system(initualconstraintsCoefficients,need)
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

                st.write(f"Múltiplo da linha {row1} somado à linha {row2} com sucesso!")
                st.write("Resumo da Regra para Resultados de Operações Elementares")
                st.write("- **Resultados Negativos**: Quando o resultado de uma operação elementar resulta em um valor negativo, você pode multiplicar toda a equação por `-1` para torná-la positiva.")
                st.write("- **Resultados Zero**:")
                st.write("  - **Se o resultado for `0 = 0`**: Isso indica que você tem uma equação redundante. Nesse caso, não precisa fazer nada, pois a equação não altera a solução do sistema.")
                st.write("  - **Se o resultado for `0 = b` (onde `b ≠ 0`)**: Isso indica uma contradição e sugere que o sistema não possui solução.")

            else:
                st.write("A operação resultou em uma linha nula, o que foi evitado.")

        # Exibir o sistema linear após a troca de linhas
        st.write("Sistema Linear após da Sistema Linear após da adição de múltiplo de uma linha à outra linha:")
        # display_system(constraints_coefficients, need)

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
    # if selected_restrictions["pausa_45_minutos"]:    
    #     for i in range(0, num_periods - 18):  # Verifica cada intervalo de 4,5 horas de condução
    #         constraint = solver.Add(solver.Sum(X[i:i+18]) <= 18)  # Máximo de 18 períodos de condução (4,5 horas)
    #         constraints.append(constraint)
    #         constraint = solver.Add(solver.Sum(X[i+18:i+21]) >= 3)  # Pausa de 3 períodos (45 minutos)
    #         constraints.append(constraint)
            
    # 6. [Ajustado] Pausa obrigatória de 45 minutos após 4,5 horas de trabalho
    if selected_restrictions["pausa_45_minutos"]:
        for start in range(num_periods - 21):  # 4,5 horas (18 períodos) + pausa (3 períodos)
            work_period = solver.Sum(X[start + k] for k in range(18))
            pause_period = solver.Sum(X[start + 18 + k] for k in range(3))
            constraint = solver.Add(work_period <= 18)
            constraints.append(constraint)
            constraint = solver.Add(pause_period >= 3)        
            constraints.append(constraint)
            
    # if (selected_restrictions["pausa_45_minutos"] or 
    #     selected_restrictions["repouso_diario_reduzido"] or 
    #     selected_restrictions["repouso_diario_minimo"] or 
    #     selected_restrictions["limite_semanal"] or
    #     selected_restrictions["limite_diario"]):
    #     # 1. Validação da demanda
    #     max_periods_per_worker = 224  # 56 horas por semana em períodos de 15 minutos
    #     total_need = sum(need)
    #     min_workers_required = ceil(total_need / max_periods_per_worker)
        
    #     msgResult.append(f"Validação inicial:")
    #     msgResult.append(f"Total de períodos de necessidade: {total_need}")
    #     msgResult.append(f"Capacidade máxima de um trabalhador por semana: {max_periods_per_worker}")
    #     msgResult.append(f"Número mínimo de trabalhadores necessários (estimado): {min_workers_required}")

    #     # Configuração inicial baseada na Regulamentação 561/2006
    #     max_hours_per_week = 56  # Máximo de 56 horas de condução por semana
    #     max_hours_per_fortnight = 90  # Máximo de 90 horas de condução por duas semanas

    #     # Cálculo do limite semanal e quinzenal em períodos de 15 minutos
    #     max_periods_per_worker = max_hours_per_week * 4  # Limite semanal (4 períodos por hora)
    #     max_periods_per_fortnight = max_hours_per_fortnight * 4  # Limite quinzenal (4 períodos por hora)
        
    #     msgResult.append(f"Capacidade máxima semanal ajustada: {max_periods_per_worker} períodos ({max_hours_per_week} horas/semana)")
    #     msgResult.append(f"Capacidade máxima quinzenal ajustada: {max_periods_per_fortnight} períodos ({max_hours_per_fortnight} horas/quinzena)")

    #     # Calcular se a demanda pode ser atendida
    #     total_need = sum(need)
    #     if total_need > max_periods_per_fortnight:
    #         msgResult.append("A demanda não pode ser atendida com base nos limites da regulamentação 561/2006.")
    #         return None, None, None, None, None, None, None, None, msgResult  # Ou outra forma de lidar com a inviabilidade
    #     else:
    #         msgResult = None
    # else:
    msgResult = None        

    solver.EnableOutput()

    # Dentro da sua função solve_shift_schedule
    finalDensity = calculate_density(constraints_coefficients)
    # Calcular densidade
    initialDensity = calculate_density(initualconstraintsCoefficients)
    
    status = None
    total_workers = None
    workers_schedule = None
    
    # Inicializando uma lista para armazenar os resultados
    statisticsResult = []
    
    if finalDensity <= densidadeAceitavel:
        # Resolver o problema
        status = solver.Solve()
        
        
        # Iterar para resolver conflitos
        max_iterations = 5
        relaxation_level = 1  # Relaxa as restrições progressivamente
        iteration = 0
        while status != pywraplp.Solver.OPTIMAL and iteration < max_iterations:
            print(f"Iteração {iteration + 1}: Conflito detectado. Relaxando restrições...")
    
            # Relaxar restrições relacionadas
            if selected_restrictions["limite_diario"]:
                relax_restrictions(solver, constraints, relaxation_level)
            if selected_restrictions["limite_semanal"]:
                relax_restrictions(solver, constraints, relaxation_level)
            if selected_restrictions["repouso_diario_minimo"]:
                relax_restrictions(solver, constraints, relaxation_level)
    
    
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
                workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
                total_workers = solver.Objective().Value()

            # Resolver novamente
            status = solver.Solve()
            iteration += 1


            # return 0, [], constraints_coefficients, 0, ["Estado do Modelo: Solução ótima não encontrada"]  # Evita retornar None

        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Total de trabalhadores necessários: {total_workers}")

        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Tipo do Modelo: {tipo_modelo(solver)}")

        # Adicionando estatísticas do solver aos resultados
        statisticsResult.append(f"Tempo total de resolução: {solver.wall_time()} ms")
        statisticsResult.append(f"Número total de iterações: {solver.iterations()}")
        statisticsResult.append(f"Número de restrições: {solver.NumConstraints()}")
        statisticsResult.append(f"Número de variáveis: {solver.NumVariables()}")
    else:
        statisticsResult.append(f"Estado do Modelo: Densidade não aceitável")
        workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
        total_workers = solver.Objective().Value()

        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Total de trabalhadores necessários: {total_workers}")

        # Adicionando o tipo do modelo aos resultados
        statisticsResult.append(f"Tipo do Modelo: {tipo_modelo(solver)}")

        # Adicionando estatísticas do solver aos resultados
        statisticsResult.append(f"Tempo total de resolução: {solver.wall_time()} ms")
        statisticsResult.append(f"Número total de iterações: {solver.iterations()}")
        statisticsResult.append(f"Número de restrições: {solver.NumConstraints()}")
        statisticsResult.append(f"Número de variáveis: {solver.NumVariables()}")
        
    # save_data(constraints_coefficients, 'constraints_coefficients.json')
    
    return solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult

# Interface do Streamlit
st.title("Shift Scheduler")
st.write("""Otimização de agendamento de turnos de motoristas para cobrir a necessidade de cada período de tempo ao longo de um período.""")

# Inicializa a variável default_need vazia
default_need = []
need_input = None
num_periods = None

initualconstraintsCoefficients = []
# Carrega os dados do arquivo, se existirem
initualconstraintsCoefficients = load_data('initualconstraintsCoefficients.json')

# Restrições de cobertura de necessidade em cada período
# constraints = []

# Criando o formulário
with st.form("paramModelo"):
    # Colunas do formulário
    col1, col2 = st.columns(2)
    
    # Entradas do usuário para horas e períodos na primeira coluna
    with col1:
        total_hours = st.number_input("Quantidade total (horas) | Valores Máximos: | Dia (9h/36p) | Semanal (56h/224p)| Quinzenal (90h/360per):", min_value=1, value=1)
        period_minutes = st.number_input("Duração de cada período (em minutos):", min_value=1, value=15)
        variable_type = st.selectbox("Escolha o tipo de variável:", ["Inteira", "Binária", "Contínua"])
        solverParamType = st.selectbox("Use 'GLOP' para problemas de otimização linear contínuos (LP) ou 'SCIP' para problemas de programação inteira mista (MIP)", ["SCIP", "GLOP"])
        percentual_aceitavel = st.number_input("Densidade Aceitável p/ Execução do Modelo:", min_value=0.10)
        limit_Workers = st.number_input("Limite de Trabalhadores (se 0, não limita)", min_value=0)

        # Seleção das restrições
        st.write("Restrições a serem aplicadas:")
        with st.expander("Restrições", expanded=True):
            selected_restrictions = {}
            for restriction in restrictions:
                checkbox_label = f"{restriction['Descrição']} | {restriction['Fórmula']}"  # Concatenando a descrição e a fórmula
                # Usando "Key" como chave única para a checkbox
                unique_key = restriction["Key"]
                selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label)
                
                # Verificar se a chave é "limiteDiario" para marcar como padrão
                # default_checked = restriction["Key"] == "limite_diario"
                # selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, value=default_checked)
                
    # Seleção do tipo de variável na segunda coluna
    with col2:
        fixar_valores = st.checkbox("Fixar Valores Aleatórios", value=False)

        # Condicional para definir `default_need` com valores fixos ou novos aleatórios
        if fixar_valores:
            default_need = gerar_valores_aleatorios(total_hours, period_minutes)
        else:
            default_need = np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()
        
        #Random
        # default_need = np.random.randint(0, 11, size=(total_hours * 60) // period_minutes).tolist()
        need_input = st.text_area(f"Demanda de trabalhadores (separado por vírgulas para cada período de {period_minutes} minutos):",', '.join(map(str, default_need)), height=210 )
        # need_input = st.text_area(f"Demanda de trabalhadores (separado por vírgulas para cada período de {period_minutes} minutos):",default_need, height=210 )
        #Exibir a quantidade de elementos em default_need
        need = [need.strip() for need in need_input.split(',')] 
        st.write(f"**Número de elementos em 'default_need': {len(need)}**")
        
        texto = """
            Principais Limites da Regulamentação 561/2006:

            - Limite diário de condução:
            9 horas por dia no máximo.
            
                    Exceção: Pode ser estendido para 10 horas por dia, no máximo duas vezes por semana.

            - Limite semanal de condução:
            56 horas por semana no máximo.

            - Limite quinzenal de condução:
            90 horas a cada 2 semanas no máximo.

            - Períodos de descanso:
                - Repouso diário regular: 11 horas por dia (pode ser dividido em dois períodos: um de pelo menos 3 horas e outro de pelo menos 9 horas).
                - Repouso diário reduzido: 9 horas, permitido até 3 vezes por semana.
                - Repouso semanal: 45 horas (pode ser reduzido para 24 horas uma vez a cada duas semanas).

            - Intervalos obrigatórios:
                - Após 4,5 horas de condução, é necessário um intervalo de 45 minutos.
            """
        
        st.write(texto)
        
    # Botão de submit do formulário
    submit_button = st.form_submit_button("Aplicar Parâmetros no Modelo")               
        
if submit_button:
    
    # Processamento dos dados de demanda
    need = [need.strip() for need in need_input.split(',')]
    st.write(f"**Número de elementos em 'default_need': {len(need)}**")
    
    # Calculando o número total de variáveis e restrições
    num_vars = len(need)
    num_restricoes = sum(1 for restricao in restrictions if selected_restrictions.get(restricao["Key"]))

    # Preenchendo os valores RHS com a demanda
    rhs_values = [int(value) for value in need]
    
    model_input = format_lp_output(num_vars, num_restricoes, rhs_values)
    with st.expander("Modelo em LP", expanded=False):
        st.write(f"**Modelo: {model_input}**")
            
# Estrutura de colunas para exibir resultados
col_operations, col_operations2 = st.columns(2)

num_periods = (total_hours * 60) // period_minutes
# generate_model(num_periods, variable_type, selected_restrictions, need)

# Matriz de restrições
num_periods = (total_hours * 60) // period_minutes

# Preencher a matriz de restrições de acordo com os períodos cobertos
need = list(map(int, need_input.split(',')))
num_periods = len(need)
constraints_coefficients = np.zeros((num_periods, num_periods), dtype=int)
for i in range(num_periods):
    for j in range(num_periods):
    # Se o período (i - j) está entre os intervalos definidos, marcamos como 1
        if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
            constraints_coefficients[i, j] = 1  # Atualiza com cobertura
        else:
           constraints_coefficients[i, j] = 0  # Sem cobertura

# Se não houver dados no arquivo, inicializa com uma matriz padrão
if initualconstraintsCoefficients is None:
    initualconstraintsCoefficients = constraints_coefficients
    save_data(initualconstraintsCoefficients, 'initualconstraintsCoefficients.json')  # Salva a matriz padrão no arquivo

atualizarFicheiro = st.checkbox("Forçar Atualização Ficheiro", value=False)
# Condicional para definir `default_need` com valores fixos ou novos aleatórios
if atualizarFicheiro:
    initualconstraintsCoefficients = constraints_coefficients
    save_data(initualconstraintsCoefficients, 'initualconstraintsCoefficients.json')
    save_data(constraints_coefficients, 'constraints_coefficients.json')

if num_periods > 0:
    initialDensityMatrix = calculate_density(constraints_coefficients)
    initialDensityMatrix = initialDensityMatrix #* 100
    st.write(f"Densidade do Modelo Inicial: {initialDensityMatrix:.2f}%")

msgResult = []
# Botão para gerar o modelo
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
            with st.expander("Operações", expanded=True):
                swap_rows_c = st.checkbox("Troca de Equações")
                multiply_row_c = st.checkbox("Multiplicação por Constante")
                add_multiple_rows_c = st.checkbox("Somar Múltiplo de uma Equação a Outra")
                
                if (not swap_rows_c and  
                not multiply_row_c and not add_multiple_rows_c):
                    constraints_coefficients = load_data('constraints_coefficients.json') 
                    solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers)
                    if msgResult is None:
                        save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo
                else:
                    # swap_rows_c = st.checkbox("Troca de Equações", key="swap")
                    if swap_rows_c:
                            row1 = st.number_input("Escolha a linha 1 para trocar:", min_value=0, max_value=len(constraints_coefficients)-1)
                            row2 = st.number_input("Escolha a linha 2 para trocar:", min_value=1, max_value=len(constraints_coefficients)-1)
                            swap_rows = row1, row2
                            constraints_coefficients = load_data('constraints_coefficients.json')
                            solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=swap_rows, multiply_row=None, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers)
                            if msgResult is None:
                                save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo
                    # else:
                    #     constraints_coefficients = load_data('constraints_coefficients.json') 
                    #     solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel)
                    #     if msgResult is None:
                    #         save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo
                    # multiply_row_c = st.checkbox("Multiplicação por Constante", key="multiply")
                    if multiply_row_c:
                        row = st.number_input("Escolha a linha para multiplicar:", min_value=0, max_value=len(constraints_coefficients)-1)
                        constant = st.number_input("Escolha a constante para multiplicar:", value=1)
                        if constant != 0: #and st.button("Aplicar Multiplicação"):
                            multiply_row = (row, constant)
                            constraints_coefficients = load_data('constraints_coefficients.json')
                            solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None,multiply_row=multiply_row, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers)
                            if msgResult is None:
                                save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo
                        else:
                            st.warning("A constante de multiplicação não pode ser zero!")
                    # else:
                    #     constraints_coefficients = load_data('constraints_coefficients.json')
                    #     solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None,multiply_row=None, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel)
                    #     if msgResult is None:
                    #         save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo

                    # add_multiple_rows_c = st.checkbox("Somar Múltiplo de uma Equação a Outra", key="addMultiply")
                    if add_multiple_rows_c:
                        row1 = st.number_input("Escolha a linha base para somar:", min_value=0, max_value=len(constraints_coefficients)-1, key="row1_sum")
                        row2 = st.number_input("Escolha a linha que vai receber o múltiplo:", min_value=0, max_value=len(constraints_coefficients)-1, key="row2_sum")
                        multiple = st.number_input("Escolha o múltiplo para somar:", value=0, key="multiple_sum")
                        add_multiple_rows = row1, row2, multiple
                        constraints_coefficients = load_data('constraints_coefficients.json')
                        solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=add_multiple_rows,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers)
                        if msgResult is None:
                            save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo
                    # else:
                    #     constraints_coefficients = load_data('constraints_coefficients.json')
                    #     solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel)
                    #     if msgResult is None:
                    #         save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo                    
            # Exibir resultados na primeira coluna
        with col_resultsII:
            if msgResult is None:
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
                        with st.expander("Escalonamento dos Trabalhadores", expanded=True):
                            st.write("**Escalonamento dos trabalhadores (1 significa que um trabalhador começa neste período):**")
                            st.write(workers_schedule)
                            # Visualização dos dados
                            st.bar_chart(workers_schedule)

                            # # Criando o gráfico de área
                            # plt.fill_between(range(len(workers_schedule)), workers_schedule, color="skyblue", alpha=0.5)
                            # plt.plot(workers_schedule, color="Slateblue", alpha=0.6)
                            # plt.title("Escalonamento de Trabalhadores (Gráfico de Área)")
                            # plt.xlabel("Período")
                            # plt.ylabel("Número de Trabalhadores")
                            # plt.grid(True)

                            # st.pyplot(plt)
                            
                            # Exemplo de dados fictícios (10 trabalhadores x 24 períodos)
                            # workers_schedule = np.random.randint(0, 5, (10, 24))

                            # # Criando o heatmap
                            # plt.figure(figsize=(10, 8))
                            # sns.heatmap(workers_schedule, cmap="YlGnBu", annot=True, fmt="d", cbar=True)
                            # plt.title("Mapa de Calor de Escalonamento de Trabalhadores")
                            # plt.xlabel("Período")
                            # plt.ylabel("Trabalhador")

                            # st.pyplot(plt)
                            
                            # Criando o boxplot
                            # plt.boxplot(workers_schedule)
                            # plt.title("Distribuição de Escalonamento de Trabalhadores (Boxplot)")
                            # plt.ylabel("Número de Trabalhadores")
                            # plt.grid(True)

                            # st.pyplot(plt)
                            
                            # # Criando um gráfico de linha
                            # plt.plot(range(len(workers_schedule)), workers_schedule)
                            # plt.title("Escalonamento de Trabalhadores ao Longo do Tempo")
                            # plt.xlabel("Período")
                            # plt.ylabel("Número de Trabalhadores")
                            # plt.grid(True)

                            # st.pyplot(plt)
                            
                            # # Criando o gráfico de barras
                            # plt.bar(range(len(workers_schedule)), workers_schedule)
                            # plt.title("Escalonamento de Trabalhadores por Período")
                            # plt.xlabel("Período")
                            # plt.ylabel("Número de Trabalhadores")
                            # plt.xticks(rotation=90)  # Girando os rótulos do eixo x se necessário
                            # plt.grid(True)
                            
                            # st.pyplot(plt)
                            
                            # # Criando o gráfico de barras empilhadas
                            # plt.bar(range(len(workers_schedule[0])), workers_schedule[0], label='Trabalhador 1')
                            # plt.bar(range(len(workers_schedule[1])), workers_schedule[1], bottom=workers_schedule[0], label='Trabalhador 2')
                            # plt.bar(range(len(workers_schedule[2])), workers_schedule[2], bottom=workers_schedule[0] + workers_schedule[1], label='Trabalhador 3')

                            # plt.title("Escalonamento Empilhado de Trabalhadores por Período")
                            # plt.xlabel("Período")
                            # plt.ylabel("Número de Trabalhadores")
                            # plt.legend()

                            # st.pyplot(plt)
                            
                            # # Número de períodos (ou horas)
                            # categories = [f"Período {i}" for i in range(1, len(workers_schedule) + 1)]

                            # # Criando o gráfico de radar
                            # angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                            # workers_schedule += workers_schedule[:1]  # Fechar o gráfico
                            # angles += angles[:1]

                            # fig, ax = plt.subplots(figsize=(6, 6), dpi=120, subplot_kw=dict(polar=True))
                            # ax.fill(angles, workers_schedule, color='blue', alpha=0.25)
                            # ax.plot(angles, workers_schedule, color='blue', linewidth=2)

                            # ax.set_yticklabels([])
                            # ax.set_xticks(angles[:-1])
                            # ax.set_xticklabels(categories, rotation=90)

                            # plt.title("Escalonamento de Trabalhadores (Radar)", size=16)
                            # st.pyplot(fig)
                                            
                            # # Criando o histograma
                            # plt.hist(workers_schedule, bins=range(min(workers_schedule), max(workers_schedule) + 2), color='skyblue', edgecolor='black')
                            # plt.title("Distribuição de Trabalhadores por Período (Histograma)")
                            # plt.xlabel("Número de Trabalhadores")
                            # plt.ylabel("Frequência")
                            # plt.grid(True)

                            # st.pyplot(plt) 
                            
                            # # Criando o gráfico de linha
                            # plt.plot(workers_schedule, marker='o', linestyle='-', color='b')
                            # plt.title("Escalonamento de Trabalhadores (Linha com Marcação)")
                            # plt.xlabel("Período")
                            # plt.ylabel("Número de Trabalhadores")
                            # plt.grid(True)

                            # # Adicionando marcas de evento
                            # for i, val in enumerate(workers_schedule):
                            #     if val > 0:
                            #         plt.text(i, val + 0.2, f'{val}', ha='center')

                            # st.pyplot(plt)
                            
                            # # Criando uma lista de tarefas para o gráfico de Gantt
                            # tasks = []

                            # # Gerando as tarefas de Gantt (um trabalhador começando em cada período com duração de 1 unidade)
                            # for period, workers in enumerate(workers_schedule):
                            #     for worker_id in range(workers):
                            #         tasks.append({
                            #             'Trabalhador': f'Trab. {worker_id + 1}',
                            #             'Início': period,
                            #             'Fim': period + 1,
                            #             'Período': period
                            #         })

                            # # Criando um DataFrame com as tarefas
                            # df = pd.DataFrame(tasks)

                            # # Criando o gráfico de Gantt com Plotly
                            # fig = px.timeline(df, 
                            #                 x_start="Início", 
                            #                 x_end="Fim", 
                            #                 y="Trabalhador", 
                            #                 color="Trabalhador", 
                            #                 title="Escalonamento de Trabalhadores (Gráfico de Gantt)",
                            #                 labels={"Trabalhador": "Trabalhador", "Período": "Período"},
                            #                 category_orders={"Trabalhador": df["Trabalhador"].unique().tolist()})

                            # # Ajuste para melhor visualização
                            # fig.update_yaxes(categoryorder="total ascending")
                            # fig.update_layout(xaxis_title="Período", yaxis_title="Trabalhador", showlegend=False)

                            # # Exibindo o gráfico no Streamlit
                            # st.plotly_chart(fig)                            
                            
                            # Criando o gráfico de área
                            # fig = go.Figure()
                            
                            # # Criando os rótulos personalizados para o eixo X
                            # periods = [i * 15 for i in range(len(workers_schedule))]  # De 0 a 15 * 383 minutos

                            # # Adicionando a área representando a alocação de motoristas ao longo do tempo
                            # fig.add_trace(go.Scatter(
                            #     x=np.arange(len(workers_schedule)),
                            #     y=workers_schedule,
                            #     fill='tozeroy',  # Preenchendo a área abaixo da linha
                            #     mode='none',     # Não exibe a linha
                            #     name='Alocação de Motoristas'
                            # ))

                            # # Ajustando o layout do gráfico
                            # fig.update_layout(
                            #     title="Escalonamento de Motoristas ao Longo do Tempo",
                            #     xaxis_title="Período de 15 minutos",
                            #     yaxis_title="Quantidade de Motoristas Alocados",
                            #     template="plotly_dark"
                            # )

                            # # Exibindo o gráfico no Streamlit
                            # st.plotly_chart(fig)
                                                        
                            # # Convertendo os dados para um DataFrame para facilitar a criação do heatmap
                            # df = pd.DataFrame(workers_schedule, columns=['Motoristas Alocados'])
                            # df['Período'] = df.index

                            # # Criando o Heatmap
                            # fig = px.imshow([df['Motoristas Alocados']], 
                            #                 labels={'x': 'Período', 'y': 'Motoristas'}, 
                            #                 x=df['Período'], 
                            #                 color_continuous_scale='Viridis',  # A escala de cores pode ser ajustada
                            #                 title="Heatmap da Alocação de Motoristas ao Longo do Tempo")

                            # # Exibindo o gráfico no Streamlit
                            # st.plotly_chart(fig)                            
                            
                            
                            
                            
    
    col_resultsIniI, col_resultsIniII, col_resultsIniIII = st.columns(3)
    with col_resultsIniI:
        if msgResult is None:    
            if initialDensityMatrix is not None:
                st.subheader("Matriz de Restrições")
                # Exibir a densidade
                st.write(f"**Densidade da Matriz de Restrições:** {initialDensityMatrix:.4f}")
                with st.expander("Matriz", expanded=True):
                    fig, ax = plt.subplots(figsize=(14, 8))
                    sns.heatmap(initualconstraintsCoefficients, cmap="Blues", cbar=True, ax=ax, linewidths=0.01, annot=False, fmt="d")
                    plt.title('Matriz de Restrições')
                    plt.xlabel('X')
                    plt.ylabel('Períodos')
                    st.pyplot(fig)
                st.write("A matriz de restrições mostra quais trabalhadores podem cobrir cada período.")
                    
    with col_resultsIniII:
        if msgResult is None:
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
    with col_resultsIniIII:
        if msgResult is None:
            #Exibir a densidade
            st.subheader("Matriz Coeficientes Pós Operações Elementares")
            st.write(f"**Densidade Final da Matriz Coeficientes:** {finalDensity:.4f}")
            with st.expander("Matriz", expanded=True):
                figNew, axNew = plt.subplots(figsize=(14, 8))
                constraints_coefficients = load_data('constraints_coefficients.json')
                sns.heatmap(constraints_coefficients, cmap="gray_r", cbar=True, ax=axNew, linewidths=0.01, annot=False, fmt="d")
                plt.title('Matriz de Coeficientes')
                plt.xlabel('X')
                plt.ylabel('Períodos')
                st.pyplot(figNew)

    if msgResult != None:
        # Criando o DataFrame a partir da lista msgResult
        # Processar statisticsResult para separar descrições e valores
        resultsMsgResult = {
            "Descrição": [],
            "Valor": []
        }
        # Preencher o dicionário com os resultados
        for stat in msgResult:
            # Separar a descrição e o valor usando ':'
            if ':' in stat:
                descricao, valor = stat.split(':', 1)  # Divide apenas na primeira ocorrência
                resultsMsgResult["Descrição"].append(descricao.strip())  # Adiciona a descrição sem espaços em branco
                resultsMsgResult["Valor"].append(valor.strip())  # Adiciona o valor sem espaços em branco
            else:
                # Caso não haja ':' no stat, adicionar como descrição e valor em branco
                resultsMsgResult["Descrição"].append(stat)
                resultsMsgResult["Valor"].append("")
        # Criar um DataFrame a partir do dicionário
        dfmsgResult = pd.DataFrame(resultsMsgResult)
        # Exibir o DataFrame como tabela
        st.table(dfmsgResult)

except Exception as e:
    st.error(f"Ocorreu um erro: {e}")