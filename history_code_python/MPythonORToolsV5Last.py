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

# Habilitar a largura total da página
st.set_page_config(layout="wide")

# Função para gerar valores aleatórios e armazená-los no cache
@st.cache_resource
def gerar_valores_aleatorios(total_hours, period_minutes):
    return np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()

# Definindo as restrições
elementalOperations = [
    {
        "Descrição": "Troca de Equações",
        "Fórmula": r"$E_i \leftrightarrow E_j$",
        "Detalhes": "O número de trabalhadores alocados em períodos que atendem as janelas válidas deve ser suficiente para satisfazer a necessidade mínima do período \(i\). A operação envolve trocar duas linhas do sistema de equações.",
        "Key": "troca_equacoes"
    },
    {
        "Descrição": "Multiplicação por Constante",
        "Fórmula": r"$E_i: \sum_{j} a_{ij} X_j = b_i \quad \text{torna-se} \quad E_i': \sum_{j} (k \cdot a_{ij}) X_j = k \cdot b_i$",
        "Detalhes": "A multiplicação de uma linha por uma constante é útil para simplificar valores ou cancelar termos, mas deve ser feita com cuidado para evitar um aumento indesejado no fill-in da matriz.",
        "Key": "multiplicacao_por_constante"
    },
    {
        "Descrição": "Somar Múltiplo de uma Equação a Outra",
        "Fórmula": r"$E_2: \sum_{j} a_{2j} X_j = b_2 \quad \text{torna-se} \quad E_2': \sum_{j} (a_{2j} + \alpha \cdot a_{1j}) X_j = b_2 + \alpha \cdot b_1$",
        "Detalhes": "Após no máximo 18 períodos (4,5 horas), deve haver uma pausa de 3 períodos (45 minutos). Esta operação envolve adicionar um múltiplo de uma linha a outra para manipular as equações do sistema.",
        "Key": "soma_multiplo_equacao"
    },
    {
        "Descrição": "Somar Múltiplo de uma Equação a Outra (automático)",
        "Fórmula": r"$E_2: \sum_{j} a_{2j} X_j = b_2 \quad \text{torna-se} \quad E_2': \sum_{j} (a_{2j} + \alpha \cdot a_{1j}) X_j = b_2 + \alpha \cdot b_1$",
        "Detalhes": "Após no máximo 18 períodos (4,5 horas), deve haver uma pausa de 3 períodos (45 minutos). Essa operação é feita automaticamente pelo solver para ajustar as equações sem intervenção manual.",
        "Key": "soma_multiplo_equacao_automatica"
    }
]




# Definindo as restrições
restrictions = [
    {
        "Descrição": "Cobertura Necessidade",
        "Fórmula": r"$\sum_{j \in \text{janelas válidas}(i)} X[j] \geq \text{need}[i]$",
        "Detalhes": "O número de trabalhadores alocados em períodos que atendem as janelas válidas deve ser suficiente para satisfazer a necessidade mínima do período \(i\).",
        "Key": "cobertura_necessidade"
    },
    {
        "Descrição": "Limite Diário Condução",
        "Fórmula": r"$\sum_{p \in \text{dia}} X[p] \leq 36p (1p=15mim, logo  36p=9h)$",
        "Detalhes": "Pode ser 40 períodos (10 horas) duas vezes por semana.",
        "Key": "limite_diario"
    },
    {
        "Descrição": "Pausa 45min Pós 4.5h Condução",
        "Fórmula": r"$\text{Pausa} \geq 3p (1p=15mim, logo 3p=45min)$",
        "Detalhes": "Após no máximo 18 períodos (4,5 horas), deve haver uma pausa de 3 períodos (45 minutos).",
        "Key": "pausa_45_minutos"
    },
    {
        "Descrição": "Fracionamento Pausa",
        "Fórmula": r"$\text{Pausa} | 15 min + 30 min$",
        "Detalhes": "A pausa de 3p períodos pode ser feita como 15 minutos seguidos de 30 minutos.",
        "Key": "divisao_pausa1530"
    },
        {
        "Descrição": "Fracionamento Pausa",
        "Fórmula": r"$\text{Pausa} | 30 min + 15 min$",
        "Detalhes": "A pausa de 3p períodos pode ser feita como 30 minutos seguidos de 15 minutos.",
        "Key": "divisao_pausa3015"
    },
    {
        "Descrição": "Descanso Diário Mínimo",
        "Fórmula": r"$\text{Descanso} \geq 44p (1p=15min. |  44p=11h)$",
        "Detalhes": "O motorista deve descansar pelo menos 44 períodos (11 horas) todos os dias.",
        "Key": "repouso_diario_minimo"
    },
    {
        "Descrição": "Descanso Semanal",
        "Fórmula": r"$\text{Descanso} \geq 180p (1p=15min. |  180p=45h) $",
        "Detalhes": "O motorista deve ter um período de repouso de 180 períodos (45 horas) toda semana.",
        "Key": "repouso_semanal"
    },
    {
        "Descrição": "Descanso Pós 6 dias Trabalho",
        "Fórmula": r"$\text{6 dias Trabalho} \Rightarrow \text{Repouso}$",
        "Detalhes": "O repouso semanal deve ser gozado após seis dias de trabalho consecutivos.",
        "Key": "descanso_apos_trabalho"
    },
    {
        "Descrição": "Limite Semanal Condução",
        "Fórmula": r"$\sum_{p \in \text{semana}} X[p] \leq 224p (1p=15min. |  224p=56h)$",
        "Detalhes": "Total de períodos de trabalho durante uma semana não deve ultrapassar 224 períodos.",
        "Key": "limite_semanal"
    },
    {
        "Descrição": "Limite Quinzenal Condução",
        "Fórmula": r"$\sum_{p \in \text{quinzena}} X[p] \leq 360p (1p=15min. |  360p=90h)$",
        "Detalhes": "Total de períodos de trabalho em duas semanas não deve ultrapassar 360 períodos.",
        "Key": "limite_quinzenal"
    },
    {
        "Descrição": "Descanso Diário Reduzido",
        "Fórmula": r"$\geq 36p \text{ (1p=15mim, logo  36p=9h | Máx. 3x | 14 dias)}$",
        "Detalhes": "O repouso pode ser reduzido para 36 períodos (9 horas), mas não mais do que 3 vezes em 14 dias.",
        "Key": "repouso_diario_reduzido"
    },
    {
        "Descrição": "Descanso Quinzenal",
        "Fórmula": r"$\geq 96p (1p=15mim, logo  96p=24h)$",
        "Detalhes": "O motorista deve ter um período de repouso de 96 períodos (24 horas) a cada duas semanas.",
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
        
        # Fazer o merge usando np.maximum para manter os valores 1 da matriz binária
        #merged_data = np.maximum(resized_existing, resized_new)
        
        # Atualizar os valores diretamente
        resized_existing[:data.shape[0], :data.shape[1]] = resized_new[:data.shape[0], :data.shape[1]]
        merged_data = resized_existing
        
    else:
        # Se não existem dados antigos, os novos dados são usados diretamente
        merged_data = data
    
    # Salvar a matriz mesclada no arquivo
    with open(FILENAME, 'w') as f:
        json.dump(merged_data.tolist(), f)
       
#     # Função para carregar o estado da matriz do arquivo JSON, se ele existir
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
    return "Modelo Linear ou Inteiro"

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
def solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None, densidadeAceitavel=None, limitWorkers=0,limit_Iteration=0,limit_Level_Relaxation=0):
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


    #Foco:
    #Simples e direto: reduzir o número total de motoristas alocados em todos os períodos, sem penalizar diretamente se a demanda não for atendida.
    #Limitação:
    #Não considera explicitamente a demanda não atendida como um fator a ser penalizado, então pode levar a soluções onde a alocação mínima resulta em falhas no atendimento.

    # Função objetivo: minimizar o número total de trabalhadores
    solver.Minimize(solver.Sum(X[d] for d in range(num_periods)))
   
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
        # st.write("Sistema Linear antes da troca de linhas:")
        # display_system(initualconstraintsCoefficients,need)
        row1, row2 = swap_rows
        if (0 <= row1 < len(constraints_coefficients)) and (0 <= row2 < len(constraints_coefficients)):
            # Troca de linhas na matriz de coeficientes
            constraints_coefficients[row1], constraints_coefficients[row2] = constraints_coefficients[row2], constraints_coefficients[row1]
            # Troca correspondente no vetor de resultados
            need[row1], need[row2] = need[row2], need[row1]
            # Atualização de restrições no solver
            constraints[row1], constraints[row2] = constraints[row2], constraints[row1]
            print(f"Linhas {row1} e {row2} trocadas com sucesso!")

    # A multiplicação de uma linha por uma constante é útil se for usada para cancelar ou simplificar valores, mas pode aumentar o fill-in se não for cuidadosamente controlada.
    if multiply_row is not None:
        # Exibir o sistema linear antes da troca de linhas
        # st.write("Sistema Linear antes da multiplicação de uma linha por uma constante:")
        # display_system(initualconstraintsCoefficients,need)
        row, constant = multiply_row
        if 0 <= row < len(constraints_coefficients):
            if constant != 0:
                # Multiplicação dos coeficientes por uma constante
                constraints_coefficients[row] = [c * constant for c in constraints_coefficients[row]]
                # Atualização da restrição correspondente no solver
                new_expr = solver.Sum(constraints_coefficients[row][j] * X[j] for j in range(len(X)))
                constraints[row] = solver.Add(new_expr >= constraints[row].lb() * constant)
                print(f"Linha {row} multiplicada por {constant} com sucesso!")

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
                st.write("A operação resultou em uma linha nula, o que foi evitado.")

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
            # Se o repouso não for reduzido, soma >= 180 períodos (7 dias com 24 horas de repouso)
            # Se o repouso for reduzido, soma >= 96 períodos (apenas 1 dia de repouso)
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
                    # Supondo que X[i, j] seja uma variável de decisão de 1 se for permitido trabalho, 0 se descanso
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
        
    # 6. [Ajustado] Pausa obrigatória de 45 minutos após 4,5 horas de trabalho
    # if selected_restrictions["pausa_45_minutos"]:
    #     # Para cada ponto de partida possível, verificamos se há 18 períodos de condução seguidos por 3 períodos de pausa
    #     for start in range(num_periods - 21):  # 4,5 horas de condução (18 períodos) + pausa (3 períodos)
            
    #         # Soma de 18 períodos de condução (4,5 horas)
    #         work_period = solver.Sum(X[start + k] for k in range(18))  # 18 períodos de condução
    #         # Soma de 3 períodos de pausa (45 minutos no total)
    #         pause_period = solver.Sum(X[start + 18 + k] for k in range(3))  # 3 períodos de pausa

    #         # Limita os períodos de condução a 18
    #         constraint = solver.Add(work_period <= 18)
    #         constraints.append(constraint)

    #         # Adiciona a condição de que deve haver 3 períodos de pausa (45 minutos)
    #         constraint = solver.Add(pause_period >= 3)
    #         constraints.append(constraint)

            # # Verifica se a pausa pode ser dividida em dois períodos (15 minutos + 30 minutos ou 30 minutos + 15 minutos)
            # if verifica_divisao_pausa(start, start + 1, num_periods):  # Função para verificar se o fracionamento é permitido
                
            #     # Caso 1: Pausa fracionada de 15 minutos + 30 minutos
            #     first_half_pause = solver.Sum(X[start + 18])  # 15 minutos de pausa (primeiro período de pausa)
            #     second_half_pause = solver.Sum(X[start + 19])  # 30 minutos de pausa (segundo período de pausa)
                
            #     # A primeira parte da pausa deve ser 15 minutos
            #     constraint_15_30 = solver.Add(first_half_pause >= 1)
            #     constraints.append(constraint_15_30)
                
            #     # A segunda parte da pausa deve ser 30 minutos
            #     constraint_30_15 = solver.Add(second_half_pause >= 1)
            #     constraints.append(constraint_30_15)

            #     # Caso 2: Pausa fracionada de 30 minutos + 15 minutos
            #     second_half_pause = solver.Sum(X[start + 19])  # 30 minutos de pausa (primeiro período de pausa)
            #     first_half_pause = solver.Sum(X[start + 18])  # 15 minutos de pausa (segundo período de pausa)
                
            #     # A segunda parte da pausa deve ser 30 minutos
            #     constraint_30_15 = solver.Add(second_half_pause >= 1)
            #     constraints.append(constraint_30_15)

            #     # A primeira parte da pausa deve ser 15 minutos
            #     constraint_15_30_2 = solver.Add(first_half_pause >= 1)
            #     constraints.append(constraint_15_30_2)
            
            
            
        # for start in range(num_periods - 21):  # 4,5 horas (18 períodos) + pausa (3 períodos)
        #     work_period = solver.Sum(X[start + k] for k in range(18))
        #     pause_period = solver.Sum(X[start + 18 + k] for k in range(3))
        #     constraint = solver.Add(work_period <= 18)
        #     constraints.append(constraint)
        #     constraint = solver.Add(pause_period >= 3)        
        #     constraints.append(constraint)

        # # Verifica se há 18 períodos de condução seguidos por 3 períodos de pausa
        # for start in range(num_periods - 21):  # 4,5 horas de condução (18 períodos) + pausa (3 períodos)
        #     work_period = solver.Sum(X[start + k] for k in range(18))  # 18 períodos de condução
        #     pause_period = solver.Sum(X[start + 18 + k] for k in range(3))  # 3 períodos de pausa

        #     constraint = solver.Add(work_period <= 18)  # Limite de 18 períodos de condução
        #     constraints.append(constraint)
            
        #     constraint = solver.Add(pause_period >= 3)  # Pausa de 15 minutos após 4,5h de condução
        #     constraints.append(constraint)            
            
        # # Lógica para pausa de 45 minutos
        # for start in range(num_periods - 21):  # 4,5 horas (18 períodos) + pausa (3 períodos)
        #     work_period = solver.Sum(X[start + k] for k in range(18))  # 18 períodos de condução
        #     pause_period = solver.Sum(X[start + 18 + k] for k in range(3))  # 3 períodos de pausa

        #     # Limite de 18 períodos de condução (4,5 horas)
        #     constraint = solver.Add(work_period <= 18)
        #     constraints.append(constraint)
            
        #     # Pausa de 15 minutos após 4,5h de condução
        #     constraint = solver.Add(pause_period >= 3)        
        #     constraints.append(constraint)

        #     # Verifica a possibilidade de fracionar a pausa
        #     if verifica_divisao_pausa(start, start + 1, num_periods):
        #         # Se a pausa pode ser dividida, aplicamos as restrições de fracionamento
        #         first_half_pause = solver.Sum(X[start + 18])  # 15 minutos de pausa
        #         second_half_pause = solver.Sum(X[start + 19])  # 30 minutos de pausa

        #         # Adiciona a restrição de que a primeira parte da pausa deve ser 15 minutos
        #         constraint_15_30 = solver.Add(first_half_pause >= 1)
        #         constraints.append(constraint_15_30)
                
        #         # A segunda parte deve ser de 30 minutos
        #         constraint_30_15 = solver.Add(second_half_pause >= 1)
        #         constraints.append(constraint_30_15)

        #         # Caso 2: 30 minutos + 15 minutos
        #         second_half_pause = solver.Sum(X[start + 19])  # 30 minutos de pausa
        #         first_half_pause = solver.Sum(X[start + 18])  # 15 minutos de pausa

        #         # Adiciona a restrição de que a segunda parte da pausa deve ser 30 minutos
        #         constraint_30_15 = solver.Add(second_half_pause >= 1)
        #         constraints.append(constraint_30_15)

        #         # A primeira parte deve ser de 15 minutos
        #         constraint_15_30_2 = solver.Add(first_half_pause >= 1)
        #         constraints.append(constraint_15_30_2)            
            
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
    
    if finalDensity <= densidadeAceitavel:
        # Resolver o problema
        status = solver.Solve()
        
        # Iterar para resolver conflitos
        max_iterations = limit_Iteration
        relaxation_level = limit_Level_Relaxation  # Relaxa as restrições progressivamente
        iteration = 0
        while status != pywraplp.Solver.OPTIMAL and iteration < max_iterations:
            statisticsResult.append(f"Iteração {iteration + 1}: Conflito detectado. Relaxando restrição: {'limite_diario atual 36h '} em {relaxation_level}")
            statisticsResult.append(f"Iteração {iteration + 1}: Conflito detectado. Relaxando restrição: {'limite_semanal atual (9 ou 10 horas) '} em {relaxation_level}")
            statisticsResult.append(f"Iteração {iteration + 1}: Conflito detectado. Relaxando restrição: {'repouso_diario_minimo atual 11h'} em {relaxation_level}")
    
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
            statisticsResult.append(f"Estado Modelo: OPTIMAL")
        elif status == pywraplp.Solver.FEASIBLE:
            statisticsResult.append(f"Estado Modelo: FEASIBLE")
        elif status == pywraplp.Solver.INFEASIBLE:
            statisticsResult.append(f"Estado Modelo: INFEASIBLE")
        elif status == pywraplp.Solver.UNBOUNDED:
            statisticsResult.append(f"Estado Modelo: UNBOUNDED")
        elif status == pywraplp.Solver.ABNORMAL:
            statisticsResult.append(f"Estado Modelo: ABNORMAL")
        elif status == pywraplp.Solver.MODEL_INVALID:
            statisticsResult.append(f"Estado Modelo: MODEL_INVALID")
        elif status == pywraplp.Solver.NOT_SOLVED:
            statisticsResult.append(f"Estado Modelo: NOT_SOLVED")
        else:
            statisticsResult.append(f"Estado Modelo: NOT_SOLVED")
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
        
    save_data(constraints_coefficients, 'constraints_coefficients.json')
    
    return solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult

# Adicionando a restrição de pausa fracionada (divisão da pausa)
def verifica_divisao_pausa(i, j, num_periods):
    # """
    # Verifica se a pausa de 45 minutos pode ser fracionada em 15 + 30 minutos ou 30 + 15 minutos,
    # considerando que o motorista deve ter conduzido por no máximo 4,5 horas (18 períodos).
    # """
    # # Diferença cíclica (modular) entre os períodos
    # diff = (i - j) % num_periods
    
    # # A pausa fracionada é permitida se a condução foi feita por 4,5 horas (18 períodos)
    # if diff == 18:  # Se a diferença for de 18 períodos (4h30min), o fracionamento pode ocorrer
    #     # Verifica as condições para a pausa fracionada
    #     # Permitir fracionamento: 15min + 30min ou 30min + 15min
    #     if (i + 1) % num_periods == 0 or (i + 2) % num_periods == 0:

    """
    Verifica se a pausa de 45 minutos pode ser fracionada em 15 + 30 minutos ou 30 + 15 minutos,
    considerando que o motorista deve ter conduzido por no máximo 4,5 horas (18 períodos) ou 5 horas (20 períodos).
    """
    # # Diferença cíclica (modular) entre os períodos
    # diff = (i - j) % num_periods
    
    # # A pausa fracionada é permitida se a condução foi feita por 4,5 horas (18 períodos) ou 5 horas (20 períodos)
    # if diff == 18 or diff == 20:  # Verifica se a diferença é de 18 ou 20 períodos (4h30min ou 5h)
    #     # Verifica as condições para a pausa fracionada
    #     # Permitir fracionamento: 15min + 30min ou 30min + 15min
    #     if (i + 1) % num_periods == 0 or (i + 2) % num_periods == 0:
    #         return True

    # """
    # Verifica se a pausa de 45 minutos pode ser fracionada em 15 + 30 minutos ou 30 + 15 minutos,
    # considerando que o motorista deve ter conduzido por no máximo 4,5 horas (18 períodos) ou 5 horas (20 períodos).
    # A pausa pode ser fracionada de 15+30min ou 30+15min.
    # """
    # Diferença cíclica (modular) entre os períodos
    # diff = (i - j) % num_periods
    
    # # A pausa fracionada é permitida se a condução foi feita por 4,5 horas (18 períodos) ou 5 horas (20 períodos)
    # if diff == 18 or diff == 20:  # Verifica se a diferença é de 18 ou 20 períodos (4h30min ou 5h)
    #     # Para a pausa de 45min, pode ser fracionada em 15min+30min ou 30min+15min
    #     # Verifica as condições para o fracionamento de pausa
    #     if (i + 1) % num_periods == 0 or (i + 2) % num_periods == 0:
    #         return True
        
    #     # Caso o fracionamento seja 30min+15min, a verificação vai depender da configuração dos períodos
    #     # Para isso, podemos verificar se os dois períodos seguidos são válidos para fracionamento de 30+15min
    #     if (i + 3) % num_periods == 0 or (i + 4) % num_periods == 0:
    #         return True
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
                            
  
                        # Fora dos intervalos permitidos
                        else:
                            initualconstraintsCoefficients[i, j] = 0  # Fora do intervalo permitido

                    elif restriction["Key"] == "repouso_diario_minimo":
                        # Exemplo de condição para repouso diário mínimo
                        if 0 <= diff < 44:  # Repouso diário de 11h
                            initualconstraintsCoefficients[i, j] = 1

                    elif restriction["Key"] == "repouso_diario_reduzido":
                        # Exemplo de condição para repouso diário reduzido
                        if 0 <= diff < 36:  # Repouso de 9h
                            initualconstraintsCoefficients[i, j] = 1

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
# st.subheader("Otimização de agendamento de turnos de motoristas para cobrir a necessidade de cada período de tempo ao longo de um período.")

# Inicializa a variável default_need vazia
default_need = []
need_input = None
num_periods = None

initualconstraintsCoefficients = []
# Carrega os dados do arquivo, se existirem
initualconstraintsCoefficients = load_data('initualconstraintsCoefficients.json')

# Criando o formulário
with st.form("paramModelo"):
    with st.expander("", expanded=True):    
        # Colunas do formulário
        col1, col2, col3, col4 = st.columns(4)
        
        # Entradas do usuário para horas e períodos na primeira coluna
        with col1:
            st.write("Parâmetros Globais")
            total_hours = st.number_input("Quantidade Total (hrs):", min_value=1, value=1)
            period_minutes = st.number_input("Duração Período (min):", min_value=1, value=15)
            limit_Workers = st.number_input("Limite Trabalhadores (se 0, não limita)", min_value=0)
        
        with col2:
            st.write("Parâmetros Algorítimo")
            variable_type = st.selectbox("Tipo Variável:", ["Inteira", "Binária", "Contínua"])
            solverParamType = st.selectbox("'GLOP' (LP) | 'SCIP' (MIP)", ["SCIP", "GLOP"])
            percentual_aceitavel = st.number_input("Densidade Aceitável", min_value=0.01)
            
        with col3:
            st.write("Parâmetros Iterações/Relaxamento")
            limit_Iteration = st.number_input("Limite Iterações", min_value=0, value=0)
            limit_Level_Relaxation = st.number_input("Ajuste Relaxamento Restrições", min_value=0, value=0)
        with col4:
            fixar_valores = st.checkbox("Fixar Valores", value=True)
            # Condicional para definir `default_need` com valores fixos ou novos aleatórios
            if fixar_valores:
                default_need = gerar_valores_aleatorios(total_hours, period_minutes)
            else:
                default_need = np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()
            
            # default_need = np.random.randint(0, 11, size=(total_hours * 60) // period_minutes).tolist()
            need_input = st.text_area(f"Demanda Clientes",', '.join(map(str, default_need)), height=210 )
            #Exibir a quantidade de elementos em default_need
            need = [need.strip() for need in need_input.split(',')] 
            st.write(f"Total Demanda: {len(need)}")
    
    # Seleção das restrições
    st.subheader("Restrições")
    with st.expander("Gerais", expanded=True):
        
        selected_restrictions = {}
        col1, col2, col3 = st.columns([2, 4, 5])

        # Radiobuttons na primeira coluna
        with col1:
            st.write("Opções de Pausa")
            radio_selection = st.radio(
                "Selecione a pausa", 
                options=["Nenhuma", "Pausa 45 minutos", "Divisão Pausa 15:30", "Divisão Pausa 30:15"],
                index=0,  # Inicialmente nenhuma pausa selecionada
                key="restricoes_pausas"
            )
            for restriction in restrictions:  # Primeira metade das restrições
                checkbox_label = f"{restriction['Descrição']} | {restriction['Fórmula']}"
           
                # Verificando se a restrição está relacionada a pausas
                if restriction["Key"] == "pausa_45_minutos":
                    # Se a opção de "Pausa 45 minutos" foi selecionada no radio button
                    selected_restrictions[restriction["Key"]] = (radio_selection == "Pausa 45 minutos")
                elif restriction["Key"] == "divisao_pausa1530":
                    # Se a opção "Divisão Pausa 15:30" foi selecionada no radio button
                    selected_restrictions[restriction["Key"]] = (radio_selection == "Divisão Pausa 15:30")
                elif restriction["Key"] == "divisao_pausa3015":
                    # Se a opção "Divisão Pausa 30:15" foi selecionada no radio button
                    selected_restrictions[restriction["Key"]] = (radio_selection == "Divisão Pausa 30:15")

        # Dividir as restrições entre as outras colunas
        with col2:
            for restriction in restrictions:  # Primeira metade das restrições
                checkbox_label = f"{restriction['Descrição']} | {restriction['Fórmula']}"
                
                if restriction["Key"] == "cobertura_necessidade":
                    default_checked = restriction["Key"] == "cobertura_necessidade"
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"], value=default_checked, disabled=True)
                elif restriction["Key"] == "limite_diario":
                    default_checked = restriction["Key"] == "limite_diario"
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"], value=default_checked, disabled=True)
                elif restriction["Key"] == "repouso_diario_minimo":
                    default_checked = restriction["Key"] == "repouso_diario_minimo"
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])
                elif restriction["Key"] == "repouso_diario_reduzido":
                    default_checked = restriction["Key"] == "repouso_diario_reduzido"
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])
                
                
                # elif restriction["Key"] != "pausa_45_minutos" and restriction["Key"] != "divisao_pausa1530" and restriction["Key"] != "divisao_pausa3015":
                    # selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])

        with col3:
            for restriction in restrictions:  # Segunda metade das restrições
                checkbox_label = f"{restriction['Descrição']} | {restriction['Fórmula']}"
                
                # Verificando se a restrição está relacionada a pausas
                if (restriction["Key"] != "pausa_45_minutos" 
                    and restriction["Key"] != "divisao_pausa1530"
                    and restriction["Key"] != "divisao_pausa3015"
                    and restriction["Key"] != "cobertura_necessidade"
                    and restriction["Key"] != "limite_diario"
                    and restriction["Key"] != "repouso_diario_minimo" 
                    and restriction["Key"] != "repouso_diario_reduzido"):
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])

                    # Se a opção de "Pausa 45 minutos" foi selecionada no radio button
                    # selected_restrictions[restriction["Key"]] = (radio_selection == "Pausa 45 minutos")
                # elif restriction["Key"] != "divisao_pausa1530":
                #     # Se a opção "Divisão Pausa 15:30" foi selecionada no radio button
                #     # selected_restrictions[restriction["Key"]] = (radio_selection == "Divisão Pausa 15:30")
                # elif restriction["Key"] != "divisao_pausa3015":
                #     # Se a opção "Divisão Pausa 30:15" foi selecionada no radio button
                #     # selected_restrictions[restriction["Key"]] = (radio_selection == "Divisão Pausa 30:15")
                # else:
                    # Para todas as outras restrições
                    # selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])

        # Exibir o dicionário para ver o estado das restrições
        # st.write("Restrições selecionadas:", selected_restrictions)
        
        
        
        
        
        
        
        
        
        
        # selected_restrictions = {}
        # col1, col2, col3 = st.columns([2, 4, 5])
        
        # # Radiobuttons na primeira coluna
        # with col1:
        #     st.write("Opções de Pausa")
        #     radio_selection = st.radio(
        #         "Selecione a pausa", 
        #         options=["Nenhuma", "Pausa 45 minutos", "Divisão Pausa 15:30", "Divisão Pausa 30:15"],
        #         index=0,  # Inicialmente nenhuma pausa selecionada
        #         key="restricoes_pausas"
        #     )

        # # Dividir as restrições entre as outras colunas
        # with col2:
        #     for restriction in restrictions[:len(restrictions)//2]:  # Primeira metade das restrições
        #         checkbox_label = f"{restriction['Descrição']} | {restriction['Fórmula']}"
                
        #         if restriction["Key"] == "cobertura_necessidade":
        #             default_checked = restriction["Key"] == "cobertura_necessidade"
        #             selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"], value=default_checked, disabled=True)
        #         elif restriction["Key"] == "limite_diario":
        #             default_checked = restriction["Key"] == "limite_diario"
        #             selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"], value=default_checked, disabled=True)
        #         elif restriction["Key"] != "pausa_45_minutos" and restriction["Key"] != "divisao_pausa1530" and restriction["Key"] != "divisao_pausa3015":
        #             selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])

        # with col3:
        #     for restriction in restrictions[len(restrictions)//2:]:  # Segunda metade das restrições
        #         checkbox_label = f"{restriction['Descrição']} | {restriction['Fórmula']}"
                
        #         if restriction["Key"] == "pausa_45_minutos":
        #             selected_restrictions[restriction["Key"]] = (radio_selection == "Pausa 45 minutos")
        #         elif restriction["Key"] == "divisao_pausa1530":
        #             selected_restrictions[restriction["Key"]] = (radio_selection == "Divisão Pausa 15:30")
        #         elif restriction["Key"] == "divisao_pausa3015":
        #             selected_restrictions[restriction["Key"]] = (radio_selection == "Divisão Pausa 30:15")
        #         else:
        #             selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"])

    # Botão de submit do formulário
    submit_button = st.form_submit_button("Aplicar Modelo")               
        
    if submit_button:
        
        # Processamento dos dados de demanda
        need = [need.strip() for need in need_input.split(',')]
        st.write(f"Total Demanda: {len(need)}")
        
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
with st.expander("Parâmetros", expanded=True):
    atualizarFicheiro = st.checkbox("Atualizar Ficheiro", value=False)
    if atualizarFicheiro:
        constraints_coefficients = initualconstraintsCoefficients
        save_data(initualconstraintsCoefficients, 'initualconstraintsCoefficients.json')
        save_data(constraints_coefficients, 'constraints_coefficients.json')

    if num_periods > 0:
        initialDensityMatrix = calculate_density(initualconstraintsCoefficients)
        initialDensityMatrix = initialDensityMatrix 
        st.write(f"Densidade Inicial: {initialDensityMatrix:.2f}%")

st.subheader("Operações Elementares")
# Loop para gerar as checkboxes com base nas operações
selected_operations = {}
for elementalOperation in elementalOperations:
    checkbox_label = f"{elementalOperation['Descrição']} | {elementalOperation['Fórmula']}"
    selected_operations[elementalOperation["Key"]] = st.checkbox(checkbox_label, key=elementalOperation["Key"], value=False)

# Inicializar variáveis para as operações
swap_rows = None
multiply_row = None
add_multiple_rows = None


msgResult = []
try:
    need = list(map(int, need_input.split(',')))
    # Estrutura de colunas para exibir resultados
    col_resultsI = st.columns(1)[0]
    col_resultsItI = st.columns(1)[0]

    # Exibir a matriz de restrições antes da otimização
    if len(need) != num_periods:
        st.error(f"A entrada deve ter exatamente {num_periods} valores (1 para cada período de {period_minutes} minutos).")
    else:
        with col_resultsI:
            # st.subheader("Operações Elementares")
            with st.expander("Operações", expanded=True):
                
                swap_rows_c = selected_operations["troca_equacoes"] #st.checkbox("Troca de Equações")
                multiply_row_c = selected_operations["multiplicacao_por_constante"] #st.checkbox("Multiplicação por Constante")
                add_multiple_rows_c = selected_operations["soma_multiplo_equacao"] #st.checkbox("Somar Múltiplo de uma Equação a Outra")
                add_multiple_rows_c_auto = selected_operations["soma_multiplo_equacao_automatica"] #st.checkbox("Somar Múltiplo de uma Equação a Outra - Automático")
                
                if (not swap_rows_c 
                    and not multiply_row_c 
                        and not add_multiple_rows_c
                        and not add_multiple_rows_c_auto):
                        constraints_coefficients = load_data('constraints_coefficients.json') 
                        solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation )
                else:
                    if swap_rows_c:
                        constraints_coefficients = load_data('constraints_coefficients.json') 
                        row1 = st.number_input("Escolha a linha 1 para trocar:", min_value=0, max_value=len(constraints_coefficients)-1)
                        row2 = st.number_input("Escolha a linha 2 para trocar:", min_value=1, max_value=len(constraints_coefficients)-1)
                        swap_rows = row1, row2
                        constraints_coefficients = load_data('constraints_coefficients.json')
                        solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=swap_rows, multiply_row=None, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation)
                    if multiply_row_c:
                        constraints_coefficients = load_data('constraints_coefficients.json') 
                        row = st.number_input("Escolha a linha para multiplicar:", min_value=0, max_value=len(constraints_coefficients)-1)
                        constant = st.number_input("Escolha a constante para multiplicar:", value=1)
                        if constant != 0: #and st.button("Aplicar Multiplicação"):
                            multiply_row = (row, constant)
                            constraints_coefficients = load_data('constraints_coefficients.json')
                            solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None,multiply_row=multiply_row, add_multiple_rows=None,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation)
                        else:
                            st.warning("A constante de multiplicação não pode ser zero!")
                    if add_multiple_rows_c:
                        constraints_coefficients = load_data('constraints_coefficients.json')
                        row1 = st.number_input("Escolha a linha base para somar:", min_value=0, max_value=len(constraints_coefficients)-1, key="row1_sum")
                        row2 = st.number_input("Escolha a linha que vai receber o múltiplo:", min_value=0, max_value=len(constraints_coefficients)-1, key="row2_sum")
                        multiple = st.number_input("Escolha o múltiplo para somar:", value=0, key="multiple_sum")
                        add_multiple_rows = row1, row2, multiple
                        solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=add_multiple_rows,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation)
                    if add_multiple_rows_c_auto:
                        constraints_coefficients = load_data('constraints_coefficients.json')
                        #Percorrer a matriz com o multiplicador -1, todas as linhas considerando o row2 sempre um a menor que o row1 e submetendo ao modelo.
                        for idx, row in enumerate(constraints_coefficients):
                            row1 = idx+1
                            row2 = idx
                            multiple = -1
                            add_multiple_rows = row1, row2, multiple
                            solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult = solve_shift_schedule(solverParamType, need, variable_type, constraints_coefficients, selected_restrictions, swap_rows=None, multiply_row=None, add_multiple_rows=add_multiple_rows,densidadeAceitavel=percentual_aceitavel,limitWorkers=limit_Workers,limit_Iteration=limit_Iteration,limit_Level_Relaxation=limit_Level_Relaxation)
                            # Dentro da sua função solve_shift_schedule
                            finalDensity = calculate_density(constraints_coefficients)
                            if finalDensity <= percentual_aceitavel:
                                st.write(f"Densidade final {finalDensity} atingiu o limite aceitável ({percentual_aceitavel}). Saindo do loop.")
                                break
            # Exibir resultados na primeira coluna
            with col_resultsItI:
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
                                    
                                    # Definir função para estilizar células
                                    def highlight_cell(x):
                                        """Estiliza células específicas."""
                                        df = pd.DataFrame('', index=x.index, columns=x.columns)  # Cria DataFrame vazio para estilos
                                        # Exemplo: pinta a célula onde Descrição é "Erro Grave" e Valor é -1
                                        for i, row in x.iterrows():
                                            if row["Descrição"] == "Estado Modelo" and row["Valor"] == "OPTIMAL":
                                                df.loc[i, "Valor"] = "background-color: green; color: white;"
                                            elif row["Descrição"] == "Estado Modelo" and row["Valor"] == "FEASIBLE":    
                                                df.loc[i, "Valor"] = "background-color: orange; color: white;"
                                            elif row["Descrição"] == "Estado Modelo" and (row["Valor"] == "INFEASIBLE" 
                                                                                          or row["Valor"] == "UNBOUNDED"
                                                                                          or row["Valor"] == "ABNORMAL"
                                                                                          or row["Valor"] == "MODEL_INVALID"
                                                                                          or row["Valor"] == "Densidade não aceitável"):    
                                                df.loc[i, "Valor"] = "background-color: red; color: white;"
                                            elif row["Descrição"] == "Estado Modelo" and row["Valor"] == "NOT_SOLVED":
                                                df.loc[i, "Valor"] = "background-color: blue; color: white;"
                                        return df

                                    # Aplicar estilização
                                    styled_df = results_df.style.apply(highlight_cell, axis=None)

                                    # Exibir o DataFrame estilizado
                                    # st.dataframe(styled_df)
                                    # Exibir o DataFrame como tabela
                                    st.table(styled_df)
                        
                                # Adicionando uma seção colapsável para o escalonamento
                                with st.expander("Escalonamento dos Motoristas", expanded=True):
                                    # Verificar se workers_schedule possui valores maiores que 0
                                    if workers_schedule and any(value > 0 for value in workers_schedule):
                                        # st.write("Escalonamento dos motoristas (1 significa que um motorista começa neste período):")
                                        # Converter workers_schedule para um DataFrame para visualização
                                        schedule_df = pd.DataFrame({
                                            "Período": list(range(1, len(workers_schedule) + 1)),
                                            "Motorista": workers_schedule
                                        })
                                        
                                        # Exibir tabela e gráfico
                                        # st.write("Escalonamento dos Motoristas (1 indica início de motorista em um período):")
                                        # st.table(schedule_df)
                                        st.bar_chart(schedule_df.set_index("Período"))
                                    else:
                                        st.warning("O escalonamento dos motoristas está vazio ou não contém valores válidos.")
    # Criando o formulário
    # with st.form("Exibir Gráficos"): 
    #     # Botão de submit do formulário
    #     submit_buttonGrafico = st.form_submit_button("Exibir Gráficos")               
        
    # if submit_buttonGrafico:   
        col_resultsIniI, col_resultsIniII = st.columns(2)
        with col_resultsIniI:
            if msgResult is None:    
                if initialDensityMatrix is not None:
                    st.subheader("Matriz de Restrições")
                    # Exibir a densidade
                    st.write(f"Densidade: {initialDensityMatrix:.4f}")
                    with st.expander("Matriz", expanded=True):
                        fig, ax = plt.subplots(figsize=(14, 8))
                        sns.heatmap(initualconstraintsCoefficients, cmap="Blues", cbar=False, annot=False, fmt="d", annot_kws={"size": 7})
                        plt.title('Matriz de Restrições')
                        plt.xlabel('X')
                        plt.ylabel('Períodos')
                        st.pyplot(fig)
                    st.write("A matriz de restrições mostra quantos motoristas podem cobrir cada período.")
        with col_resultsIniII:
            if msgResult is None:
                #Exibir a densidade
                st.subheader("Matriz Restrições Final")
                st.write(f"Densidade Final Matriz Restrições: {finalDensity:.4f}")
                with st.expander("Matriz", expanded=True):
                    figNew, axNew = plt.subplots(figsize=(14, 8))
                    constraints_coefficients = load_data('constraints_coefficients.json')
                    sns.heatmap(constraints_coefficients, cmap="Oranges", cbar=False, annot=False, fmt="d", annot_kws={"size": 6})
                    plt.title('Matriz Restrições')
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