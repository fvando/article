import streamlit as st
from ortools.linear_solver import pywraplp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tempfile  # Para criar arquivos temporários
import io
import sys

# Habilitar a largura total da página
st.set_page_config(layout="wide")

    
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

# Função de otimização (baseada no código anterior)
def solve_shift_scheduleOld(need, variable_type, constraint_matrix, selected_restrictions):
    num_periods = len(need)
    
    # Criar o solver
    solver = pywraplp.Solver.CreateSolver('SCIP',)  # Usar 'GLOP' para LP ou 'SCIP' para MIP
    
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

    # Restrições de cobertura de necessidade em cada período
    for i in range(num_periods):
        solver.Add(solver.Sum(X[j] for j in range(num_periods) 
                               if (0 <= (i - j) % num_periods < 18)  
                               or (19 <= (i - j) % num_periods < 23)  
                               or (25 <= (i - j) % num_periods < 43)) >= need[i])

    # 1. Limite diário de condução (9 horas ou 10 horas duas vezes por semana)
    # 9 horas = 36 períodos de 15 minutos
    # 10 horas = 40 períodos de 15 minutos, permitido 2 vezes por semana
    if selected_restrictions["limite_diario"]:    
        for day in range(num_periods // 96):  # 96 períodos de 15 minutos em um dia
            solver.Add(solver.Sum(X[day * 96 + p] for p in range(96)) <= 36)  # Limite normal de 9 horas (36 períodos)
    
    # Adiciona exceção para 10 horas de condução (40 períodos), no máximo 2 vezes por semana
    if selected_restrictions["limite_semanal"]:    
        extended_hours_days = 2  # Número de dias que pode trabalhar 10 horas
        extended_hours_vars = []
        for day in range(num_periods // 96):  # 96 períodos por dia
            extended_hours = solver.BoolVar(f'extended_hours[{day}]')
            extended_hours_vars.append(extended_hours)
            solver.Add(solver.Sum(X[day * 96 + p] for p in range(96)) <= 40 * extended_hours + 36 * (1 - extended_hours))
        solver.Add(solver.Sum(extended_hours_vars) <= extended_hours_days)

    # 2. Limite semanal de condução: 56 horas por semana
    # 56 horas = 224 períodos de 15 minutos
    if selected_restrictions["limite_semanal"]:    
        for week in range(num_periods // (96 * 7)):  # Cada semana tem 96 * 7 períodos de 15 minutos
            solver.Add(solver.Sum(X[week * 96 * 7 + p] for p in range(96 * 7)) <= 224)

    # 3. Limite quinzenal de condução: 90 horas a cada 14 dias
    # 90 horas = 360 períodos de 15 minutos
    if selected_restrictions["limite_quinzenal"]:    
        for period in range(num_periods // (96 * 14)):  # 14 dias com 96 períodos por dia
            solver.Add(solver.Sum(X[period * 96 * 14 + p] for p in range(96 * 14)) <= 360)

    # 4. Repouso diário mínimo de 11 horas (44 períodos)
    # Pode ser reduzido para 9 horas (36 períodos) três vezes entre dois períodos de repouso semanais
    if selected_restrictions["repouso_diario_minimo"]:    
        daily_rest_periods = 44  # 11 horas = 44 períodos de 15 minutos
        reduced_rest_days = 3  # Número de dias que pode reduzir o repouso para 9 horas (36 períodos)
        reduced_rest_vars = []
        for day in range(num_periods // 96):  # Cada dia tem 96 períodos de 15 minutos
            reduced_rest = solver.BoolVar(f'reduced_rest[{day}]')
            reduced_rest_vars.append(reduced_rest)
            solver.Add(solver.Sum(X[day * 96:(day + 1) * 96]) <= 96 - (daily_rest_periods - 8 * reduced_rest))
        solver.Add(solver.Sum(reduced_rest_vars) <= reduced_rest_days)

    # 5. Repouso semanal de 45 horas (180 períodos), pode ser reduzido para 24 horas uma vez a cada duas semanas
    # 45 horas = 180 períodos de 15 minutos
    # 24 horas = 96 períodos de 15 minutos
    if selected_restrictions["repouso_diario_reduzido"]:    
        reduced_weekly_rest_weeks = 1  # Quantas vezes pode reduzir o repouso para 24 horas a cada duas semanas
        weekly_rest_vars = []
        for week in range(num_periods // (96 * 7)):  # Cada semana tem 96 * 7 períodos
            reduced_rest = solver.BoolVar(f'reduced_weekly_rest[{week}]')
            weekly_rest_vars.append(reduced_rest)
            solver.Add(solver.Sum(X[week * 96 * 7:(week + 1) * 96 * 7]) >= 180 * (1 - reduced_rest) + 96 * reduced_rest)
        solver.Add(solver.Sum(weekly_rest_vars) <= reduced_weekly_rest_weeks)

    # 6. Pausa de 45 minutos (3 períodos de 15 minutos) após 4,5 horas de condução (18 períodos)
    if selected_restrictions["pausa_45_minutos"]:    
        for i in range(0, num_periods - 18):  # Verifica cada intervalo de 4,5 horas de condução
            solver.Add(solver.Sum(X[i:i+18]) <= 18)  # Máximo de 18 períodos de condução (4,5 horas)
            solver.Add(solver.Sum(X[i+18:i+21]) >= 3)  # Pausa de 3 períodos (45 minutos)

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
    density = np.sum(constraint_matrix) / (num_periods ** 2)
    
    # Adicionando o tipo do modelo aos resultados
    statisticsResult.append(f"Total de trabalhadores necessários: {total_workers}")

    # Adicionando o tipo do modelo aos resultados
    statisticsResult.append(f"Tipo do Modelo: {tipo_modelo(solver)}")

    # Adicionando estatísticas do solver aos resultados
    statisticsResult.append(f"Tempo total de resolução: {solver.wall_time()} ms")
    statisticsResult.append(f"Número total de iterações: {solver.iterations()}")
    statisticsResult.append(f"Número de restrições: {solver.NumConstraints()}")
    statisticsResult.append(f"Número de variáveis: {solver.NumVariables()}")
    statisticsResult.append(f"Variáveis: {solver.variables()}")
    statisticsResult.append(f"Constraints: {solver.Constraint().Lb()}")
    statisticsResult.append(f"Constraints: {solver.Constraint().Ub()}")
    statisticsResult.append(f"Número de constraints: {solver.constraints()}")
    statisticsResult.append(f"Número de nós: {solver.nodes()}")
    statisticsResult.append(f"Número de nós: {solver.EnableOutput()}")
    
    return solver, status, total_workers, workers_schedule, constraint_matrix, density, statisticsResult


def calculate_density(constraints_coefficients, num_periods):
    total_non_zero = 0
    total_elements = len(constraints_coefficients) * num_periods  # Total de elementos na matriz
    
    for coeffs in constraints_coefficients:
        total_non_zero += sum(1 for c in coeffs if c != 0)  # Conta elementos não nulos
    
    density = total_non_zero / total_elements if total_elements > 0 else 0  # Evita divisão por zero
    return density



# Função de otimização (baseada no código anterior)
def solve_shift_schedule(need, variable_type, constraint_matrix, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None):
    num_periods = len(need)
    
    # Criar o solver
    solver = pywraplp.Solver.CreateSolver('SCIP')  # Usar 'GLOP' para LP ou 'SCIP' para MIP
   
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

    # Restrições de cobertura de necessidade em cada período
    constraints = []

    for i in range(num_periods):
        constraint = solver.Add(solver.Sum(X[j] for j in range(num_periods) 
                               if (0 <= (i - j) % num_periods < 18)  
                               or (19 <= (i - j) % num_periods < 23)  
                               or (25 <= (i - j) % num_periods < 43) ) >= need[i])
        constraints.append(constraint)
        
    # Restrições de cobertura de necessidade em cada período
    #for i in range(num_periods):
    #    coeffs = [1] * num_periods  # Coeficientes (aqui é 1 para todos X)
        #constraint = solver.Add(solver.Sum(coeffs[j] * X[j] for j in range(num_periods)) >= need[i])
        #constraints.append(constraint)  # Armazenar a restrição
    #    constraints_coefficients.append(coeffs)  # Armazenar os coeficientes      
        
    for i in range(num_periods):
        coeffs = []
        for j in range(num_periods):
            if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
                coeffs.append(1)
            else:
                coeffs.append(0)  # Ou outro valor conforme necessário
        constraints_coefficients.append(coeffs)    
     
    # Aplicar operações elementares, se selecionadas
    #swap_rows = (0, 1)  # Exemplo: trocar as linhas 0 e 1
    #if swap_rows is not None:
    #    row1, row2 = swap_rows
        # Trocar as restrições
    #    constraints[row1], constraints[row2] = constraints[row2], constraints[row1]
    #    print(f"Linhas {row1} e {row2} trocadas com sucesso!")

    #swap_rows = (0, 1)  # Exemplo: trocar as linhas 0 e 1
    if swap_rows is not None:
        row1, row2 = swap_rows
        # Verifique se as linhas estão dentro do intervalo
        if (0 <= row1 < len(constraints)) and (0 <= row2 < len(constraints)):
            # Trocar as restrições
            constraints[row1], constraints[row2] = constraints[row2], constraints[row1]
        
            # Verifique o tamanho da lista de coeficientes
            print(f"Tamanho de constraints_coefficients: {len(constraints_coefficients)}")
        
            # Também trocar os coeficientes associados
        if (0 <= row1 < len(constraints_coefficients)) and (0 <= row2 < len(constraints_coefficients)):
            constraints_coefficients[row1], constraints_coefficients[row2] = constraints_coefficients[row2], constraints_coefficients[row1]
            print(f"Linhas {row1} e {row2} trocadas com sucesso!")
        else:
            print(f"Erro: Índices de coeficientes fora do intervalo. row1: {row1}, row2: {row2}, total de coeficientes: {len(constraints_coefficients)}")
    #else:
    #    print(f"Erro: Índices de linha fora do intervalo. row1: {row1}, row2: {row2}, total de restrições: {len(constraints)}")
        
    if multiply_row is not None:
        row, constant = multiply_row
        if (0 <= row < len(constraints)):
            if constant != 0:
                # Verificar se constraints_coefficients possui os coeficientes para a linha
                #if row < len(constraints_coefficients):
                    new_coeffs = [c * constant for c in constraints_coefficients[row]]
                    #new_constraint = solver.Add(solver.Sum(new_coeffs[j] * X[j] for j in range(num_periods)) >= constraints[row].lb() * constant)
                    # Atualizar a restrição existente
                    constraints[row] = solver.Add(solver.Sum(new_coeffs[j] * X[j] for j in range(num_periods)) >= constraints[row].lb() * constant)
                    # Atualizar os coeficientes
                    constraints_coefficients[row] = new_coeffs
                    #constraints.append(new_constraint)  # Adicionar nova restrição
                    #constraints_coefficients.append(new_coeffs)  # Armazenar novos coeficientes
                    print(f"Linha {row} multiplicada por {constant} com sucesso!")
                #else:
                #    print(f"Erro: Índice de coeficientes fora do intervalo. row: {row}, total de coeficientes: {len(constraints_coefficients)}")
        else:
            print(f"Erro: Índice da linha de multiplicação fora do intervalo. row: {row}, total de restrições: {len(constraints)}")

    #if multiply_row is not None:
    #    row, constant = multiply_row
    #    if constant != 0:
            # Multiplicar a restrição por uma constante
    #        new_expr = constraints[row].expr() * constant
    #        solver.Add(new_expr >= constraints[row].lb())  # Define nova restrição
    #        print(f"Linha {row} multiplicada por {constant} com sucesso!")

    #if add_multiple_rows is not None:
    #    row1, row2, multiple = add_multiple_rows
        # Adicionar múltiplo da linha row1 à linha row2
    #    new_expr = constraints[row2].expr() + multiple * constraints[row1].expr()
    #    solver.Add(new_expr >= constraints[row2].lb())  # Define nova restrição
    #    print(f"Múltiplo da linha {row1} somado à linha {row2} com sucesso!")

    # Se você quiser adicionar múltiplos de uma restrição a outra
    #add_multiple_rows = (1, 2, 3)  # Exemplo: adicionar 3 vezes a linha 1 à linha 2
    if add_multiple_rows is not None:
        row1, row2, multiple = add_multiple_rows
        
        # Criar nova expressão a partir dos coeficientes armazenados
        new_coeffs = [constraints_coefficients[row2][j] + multiple * constraints_coefficients[row1][j] for j in range(num_periods)]
        #new_constraint = solver.Add(solver.Sum(new_coeffs[j] * X[j] for j in range(num_periods)) >= constraints[row2].lb())
        
        # Atualizar a restrição existente
        constraints[row2] = solver.Add(solver.Sum(new_coeffs[j] * X[j] for j in range(num_periods)) >= constraints[row2].lb())
    
        # Atualizar os coeficientes
        constraints_coefficients[row2] = new_coeffs
    
        #constraints.append(new_constraint)  # Adicionar nova restrição
        #constraints_coefficients.append(new_coeffs)  # Armazenar novos coeficientes
        print(f"Múltiplo da linha {row1} somado à linha {row2} com sucesso!")

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
    densityOld = calculate_density(constraint_matrix, num_periods)
    #densityOld = np.sum(constraint_matrix) / (num_periods ** 2)
    
    # Dentro da sua função solve_shift_schedule
    densityNew = calculate_density(constraints_coefficients, num_periods)
    # Calcular densidade
    #densityNew = np.sum(constraints_coefficients) / (num_periods ** 2)
    
    # Adicionando o tipo do modelo aos resultados
    statisticsResult.append(f"Total de trabalhadores necessários: {total_workers}")

    # Adicionando o tipo do modelo aos resultados
    statisticsResult.append(f"Tipo do Modelo: {tipo_modelo(solver)}")

    # Adicionando estatísticas do solver aos resultados
    statisticsResult.append(f"Tempo total de resolução: {solver.wall_time()} ms")
    statisticsResult.append(f"Número total de iterações: {solver.iterations()}")
    statisticsResult.append(f"Número de restrições: {solver.NumConstraints()}")
    statisticsResult.append(f"Número de variáveis: {solver.NumVariables()}")
    


    return solver, status, total_workers, workers_schedule, constraint_matrix, constraints_coefficients, densityOld, densityNew, statisticsResult

# Interface do Streamlit
st.title("Shift Scheduler")
st.write("""Otimização de agendamento de turnos de motoristas para cobrir a necessidade de cada período de tempo ao longo de um período.""")

# Inicializa a variável default_need vazia
default_need = []
need_input = None
num_periods = None
constraints_coefficients = []

col1, col2 = st.columns(2)
# Entradas do usuário para horas e períodos na primeira coluna
with col1:
    total_hours = st.number_input("Quantidade total (em horas):", min_value=1, value=24)
    period_minutes = st.number_input("Duração de cada período (em minutos):", min_value=1, value=15)
    variable_type = st.selectbox("Escolha o tipo de variável:", ["Inteira", "Binária", "Contínua"])
    
    # Seleção das restrições
    st.write("#### Restrições a serem aplicadas:")
    with st.expander("Restrições", expanded=False):
        selected_restrictions = {}
        for restriction in restrictions:
            checkbox_label = f"{restriction['Descrição']} | {restriction['Fórmula']}"  # Concatenando a descrição e a fórmula
            selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label)
        
# Seleção do tipo de variável na segunda coluna
with col2:
    #if need_input is None:
    default_need = "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"
    #np.random.randint(0, 11, size=(total_hours * 60) // period_minutes).tolist()
    #"6,5,10,2" #np.random.randint(0, 11, size=(total_hours * 60) // period_minutes).tolist()
    need_input = st.text_area(f"Demanda de trabalhadores (separado por vírgulas para cada período de {period_minutes} minutos):", default_need, height=210 )
    #need_input = st.text_area(f"Demanda de trabalhadores (separado por vírgulas para cada período de {period_minutes} minutos):",', '.join(map(str, default_need)), height=210 )
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
        
    # Botão de submissão para aplicar as mudanças
    #submit_button = st.form_submit_button(label="Aplicar")

# Estrutura de colunas para exibir resultados
col_operations, col_operations2 = st.columns(2)

#with col_operations:
# Adiciona um botão para povoar a demanda de trabalhadores
#if st.button("Gerar demanda aleatória"):
#    default_need = np.random.randint(0, 11, size=(total_hours * 60) // period_minutes).tolist()
    # Exibir a quantidade de elementos em default_need
#    st.write(f"**Número de elementos em 'default_need': {len(need)}**")

# Botão para gerar o modelo
#if st.button("Gerar Modelo"):
#num_periods = (total_hours * 60) // period_minutes
#    generate_model(num_periods, variable_type, selected_restrictions, need)

# Matriz de restrições
#num_periods = (total_hours * 60) // period_minutes
#constraint_matrix = np.zeros((num_periods, num_periods), dtype=int)
# Preencher a matriz de restrições de acordo com os períodos cobertos
#    need = list(map(int, need_input.split(',')))
#    num_periods = len(need)
#    for i in range(num_periods):
#            for j in range(num_periods):
#                if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
#                    constraint_matrix[i, j] = 1  # Atualizando a matriz de restrições com os valores de cobertura


    # Criando um formulário para conter a seleção de tipo de variável
    #with st.form(key="my_form2"):
# Aplicar operações elementares à matriz de restrições **antes** da execução
#constraint_matrix = apply_elementary_operations(constraint_matrix)
   
    # Botão de submissão para aplicar as mudanças
#    submit_button2 = st.form_submit_button(label="Aplicar")

# Botão para gerar o modelo
#if st.button("Executar Modelo"):
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
            if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
                constraint_matrix[i, j] = 1  # Atualizando a matriz de restrições com os valores de cobertura

try:
    need = list(map(int, need_input.split(',')))
    
    # Estrutura de colunas para exibir resultados
    col_results, col_restrictions = st.columns(2)
    # Exibir a matriz de restrições antes da otimização
    if len(need) != num_periods:
        st.error(f"A entrada deve ter exatamente {num_periods} valores (1 para cada período de {period_minutes} minutos).")
    else:
        st.subheader("Operações Elementares nas Restrições")
        with st.expander("Operações", expanded=False):
            swap_rows = st.checkbox("Troca de Equações")
            if swap_rows:
                row1 = st.number_input("Escolha a linha 1 para trocar:", min_value=0, max_value=len(constraint_matrix)-1)
                row2 = st.number_input("Escolha a linha 2 para trocar:", min_value=0, max_value=len(constraint_matrix)-1)
                swap_rows = row1, row2
                solver, status, total_workers, workers_schedule, constraint_matrix, constraints_coefficients, densityOld, densityNew, statisticsResult = solve_shift_schedule(need, variable_type, constraint_matrix, selected_restrictions, swap_rows=swap_rows, multiply_row=None, add_multiple_rows=None)
            else:
                solver, status, total_workers, workers_schedule, constraint_matrix, constraints_coefficients, densityOld, densityNew, statisticsResult = solve_shift_schedule(need, variable_type, constraint_matrix, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None)

            multiply_row_c = st.checkbox("Multiplicação por Constante")
            if multiply_row_c:
                row = st.number_input("Escolha a linha para multiplicar:", min_value=0, max_value=len(constraint_matrix)-1)
                constant = st.number_input("Escolha a constante para multiplicar:", value=1.0)
                if constant != 0: #and st.button("Aplicar Multiplicação"):
                    multiply_row = (row, constant)
                    solver, status, total_workers, workers_schedule, constraint_matrix, constraints_coefficients, densityOld, densityNew, statisticsResult = solve_shift_schedule(need, variable_type, constraint_matrix, selected_restrictions, swap_rows=None,multiply_row=multiply_row, add_multiple_rows=None)
                else:
                    st.warning("A constante de multiplicação não pode ser zero!")
            else:
                solver, status, total_workers, workers_schedule, constraint_matrix, constraints_coefficients, densityOld, densityNew, statisticsResult = solve_shift_schedule(need, variable_type, constraint_matrix, selected_restrictions, swap_rows=None,multiply_row=None, add_multiple_rows=None)

            add_multiple_rows_c = st.checkbox("Somar Múltiplo de uma Equação a Outra")
            if add_multiple_rows_c:
                row1 = st.number_input("Escolha a linha base para somar:", min_value=1, max_value=len(constraint_matrix)-1, key="row1_sum")
                row2 = st.number_input("Escolha a linha que vai receber o múltiplo:", min_value=2, max_value=len(constraint_matrix)-1, key="row2_sum")
                multiple = st.number_input("Escolha o múltiplo para somar:", value=3.0, key="multiple_sum")
                add_multiple_rows = row1, row2, multiple
                solver, status, total_workers, workers_schedule, constraint_matrix, constraints_coefficients, densityOld, densityNew, statisticsResult = solve_shift_schedule(need, variable_type, constraint_matrix, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=add_multiple_rows)
            else:
                solver, status, total_workers, workers_schedule, constraint_matrix, constraints_coefficients, densityOld, densityNew, statisticsResult = solve_shift_schedule(need, variable_type, constraint_matrix, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None)
            # Exibir resultados na primeira coluna
            with col_results:
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
                    
        # Exibir a matriz de restrições na segunda coluna
        with col_restrictions:
            st.subheader("Matriz de Restrições")
            st.write("A matriz de restrições mostra quais trabalhadores podem cobrir cada período.")
            # Exibir a densidade
            st.write(f"**Densidade da Matriz de Restrições:** {densityOld:.4f}")
            with st.expander("Matriz", expanded=True):
                fig, ax = plt.subplots(figsize=(16, 8))
                sns.heatmap(constraint_matrix, cmap="Blues", cbar=True, ax=ax, linewidths=1.0)
                plt.title('Matriz de Restrições')
                plt.xlabel('X')
                plt.ylabel('Períodos')
                st.pyplot(fig)
            
            #Exibir a densidade
            st.write(f"**Densidade da Matriz Coeficientes:** {densityNew:.4f}")
            with st.expander("Matriz", expanded=True):
                figNew, axNew = plt.subplots(figsize=(16, 10))
                sns.heatmap(constraints_coefficients, cmap="Blues", cbar=True, ax=axNew, linewidths=0.5)
                plt.title('Matriz de Coeficientes')
                plt.xlabel('X')
                plt.ylabel('Períodos')
                st.pyplot(figNew)
except Exception as e:
    st.error(f"Ocorreu um erro: {e}")


