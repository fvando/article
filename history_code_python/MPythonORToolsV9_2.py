import html
import math
import traceback
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
import io, contextlib

from solver.heuristic import greedy_initial_allocation
from solver.lns import run_lns
import html

# Habilitar a largura total da página
st.set_page_config(layout="wide")


# Função para gerar valores aleatórios e armazená-los no cache
@st.cache_resource
def gerar_valores_aleatorios(total_hours, period_minutes):
    return np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()

def make_formula_safe(formula: str) -> str:
    """
    Converte fórmulas matemáticas para uma versão segura para Streamlit,
    evitando erros de parsing markdown como 'unmatched )'.
    Produz um texto totalmente seguro sem quebrar a UI.
    """
    if not formula:
        return ""

    safe = html.escape(formula)

    # Remove interpretações de markdown perigoso
    safe = safe.replace("|", " | ")

    # Evitar que colchetes virem links
    safe = safe.replace("[", "(").replace("]", ")")

    # Evitar inline LaTeX
    safe = safe.replace("$", "")

    # Operadores unicode → ASCII seguro
    safe = safe.replace("⋅", "*").replace("·", "*").replace("→", "->")

    # Escapar parênteses para evitar math parsing
    safe = safe.replace("(", "&#40;").replace(")", "&#41;")

    return safe


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

# ============================================================
# Função controladora para os três modos: Exact, Heuristic, LNS
# ============================================================
def run_solver_with_mode(
    mode,
    need,
    variable_type,
    constraints_coefficients,
    selected_restrictions,
    solver_param_type,
    densidade_aceitavel,     
    limit_workers,
    limit_iteration,
    limit_level_relaxation,
    max_demands_per_driver,
    tolerance_demands,
    penalty,
    swap_rows=None, 
    multiply_row=None, 
    add_multiple_rows=None,
    radio_selection_object=None 
):

    # 1) Sempre gerar a heurística inicial
    initial_allocation = greedy_initial_allocation(
        need=need,
        limit_workers=limit_workers,
        max_demands_per_driver=max_demands_per_driver,
    )
    
    
    # ======================================================
    # 🔵 FUNÇÃO AUXILIAR: Retorno padronizado para os 3 modos
    # ======================================================
    def make_return(
        solver=None,
        status="NOT_SOLVED",
        total_workers=0,
        workers_schedule=None,
        constraints_coeffs=None,
        initial_density=None,
        final_density=None,
        statistics=None,
        msg=None,
        iterations=None,
        allocation=None,
        logs=None,
    ):
        # Garante estruturas mínimas
        if statistics is None:
            statistics = []
        if msg is None:
            msg = []
        if iterations is None:
            iterations = []
        if logs is None:
            logs = {"stdout": "", "stderr": ""}

        return (
            solver,             # 1
            status,             # 2
            total_workers,      # 3
            workers_schedule,   # 4
            constraints_coeffs, # 5
            initial_density,    # 6
            final_density,      # 7
            statistics,         # 8
            msg,                # 9
            iterations,         #10
            allocation,         #11
            logs                #12
        )
    
    
    
    # ======================================================
    # MODO EXATO
    # ======================================================
    if mode == "Exact":
        (
            solver,
            status,
            total_workers,
            workers_schedule,
            constraints_coefficients_out,
            initial_density,
            final_density,
            statistics_result,
            msg,
            iterations_data,
            matrix_allocation,
            solver_logs,
        ) = solve_shift_schedule(
            solver_param_type,
            need,
            variable_type,
            constraints_coefficients,
            selected_restrictions,
            swap_rows,
            multiply_row,
            add_multiple_rows,
            densidade_aceitavel,
            limit_workers,
            limit_iteration,
            limit_level_relaxation,
            max_demands_per_driver,
            tolerance_demands,
            penalty,
            initial_allocation=None,
            fixed_assignments=None,
            radio_selection_object=radio_selection_object,
            mode="Exact"
        )

        return make_return(
            solver,
            status,
            total_workers,
            workers_schedule,
            constraints_coefficients_out,
            initial_density,
            final_density,
            statistics_result,
            msg,
            iterations_data,
            matrix_allocation,
            solver_logs
        )
        
    # ======================================================
    # MODO HEURÍSTICO
    # ======================================================
   
    if mode == "Heuristic":
        matrix_allocation = initial_allocation
        total_workers = int(matrix_allocation.sum())
        workers_schedule = list(np.sum(matrix_allocation, axis=1))
        
        statistics_result = [
            "Model State: HEURISTIC",
            f"Total workers needed: {total_workers}",
            "Model Type: Heuristic (greedy initial allocation)"
        ]

        # Para manter a mesma assinatura da função (12 retornos),
        # mas sem erro de variável não definida:
        # solver_logs =  {"stdout": "", "stderr": ""}   # nenhum log específico no modo heurístico

        return make_return(
            solver=None,
            status="HEURISTIC",
            total_workers=total_workers,
            workers_schedule=workers_schedule,
            constraints_coeffs=constraints_coefficients,
            initial_density=None,
            final_density=None,
            statistics=statistics_result,
            msg=[],
            iterations=[],
            allocation=matrix_allocation,
            logs={"stdout": "Heuristic mode executed.", "stderr": ""}
        )   

    # ======================================================
    # MODO LNS (MATHEURÍSTICO)
    # ======================================================
   
    if mode == "LNS":
        best_solution, info = run_lns(
            initial_solution=initial_allocation,
            need=need,
            variable_type=variable_type,
            constraints_coefficients=constraints_coefficients,
            selected_restrictions=selected_restrictions,
            solver_param_type=solver_param_type,
            limit_workers=limit_workers,
            limit_iteration=limit_iteration,
            limit_level_relaxation=limit_level_relaxation,
            max_demands_per_driver=max_demands_per_driver,
            tolerance_demands=tolerance_demands,
            penalty=penalty,
            max_lns_iterations=5,
            solve_fn=solve_shift_schedule
        )

        total_workers = int(best_solution.sum())
        workers_schedule = list(np.sum(best_solution, axis=1))

        statistics_result = [
            "Model State: LNS",
            f"Total workers needed: {total_workers}",
            "Model Type: Matheuristic (LNS + MILP)"
        ]

        # Para manter a mesma assinatura da função (12 retornos),
        # mas sem erro de variável não definida:
        # solver_logs = {"stdout": "", "stderr": ""}  # nenhum log específico no modo heurístico        

        return make_return(
            solver=None,
            status="LNS",
            total_workers=total_workers,
            workers_schedule=workers_schedule,
            constraints_coeffs=constraints_coefficients,
            initial_density=None,
            final_density=None,
            statistics=statistics_result,
            msg=[],
            iterations=info.get("history", []),
            allocation=best_solution,
            logs={"stdout": "LNS executed successfully.", "stderr": ""}
        )

# Função para relaxar restrições dinamicamente em caso de conflitos
def relax_restrictions(solver, constraints, relaxation_level):
    for constraint in constraints:
        # Ajusta o limite superior ou inferior das restrições para relaxar o problema
        if constraint.Lb() is not None:
            constraint.SetLb(constraint.Lb() - relaxation_level)
        if constraint.Ub() is not None:
            constraint.SetUb(constraint.Ub() + relaxation_level)

def relax_objective(solver, Y, num_periods, limit_workers, penalty, max_penalty, relaxation_level, need, tolerance_demands,  max_demands_per_driver,):
    """
    Relaxar a função objetivo, diminuindo a penalização de demandas conforme o relaxamento aumenta.
    """
    # Ajustar a penalização progressivamente
    relaxed_penalty = max(0, penalty - relaxation_level)  # Diminui a penalização até zero
    
    # Construir a função objetivo com penalização relaxada
    objective = solver.Minimize(
        solver.Sum(Y[d, t] for d in range(num_periods) for t in range(limit_workers))  # Minimizar motoristas
        + relaxed_penalty * solver.Sum(
            (need[d] * tolerance_demands - solver.Sum(Y[d, t] * max_demands_per_driver for t in range(limit_workers))) 
            for d in range(num_periods)
        )
    )
    
    return objective

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
def solve_shift_schedule(
    solver_param_type, 
    need, 
    variable_type, 
    constraints_coefficients, 
    selected_restrictions, 
    swap_rows=None, 
    multiply_row=None, 
    add_multiple_rows=None, 
    densidade_aceitavel=None, 
    limit_workers=0,
    limit_iteration=0,
    limit_level_relaxation=0, 
    max_demands_per_driver=1,
    tolerance_demands=0.01, 
    penalty = 0.01,
    initial_allocation=None,      # <<< NOVO
    fixed_assignments=None,       # <<< NOVO
    radio_selection_object=None,  # <<< NOVO
    mode="Exact"
):
    
    # Tabela de Restrições e Interdependências
    # Restrição	Descrição	Interdependência Potencial
    # Cobertura de Necessidade	Garante que a demanda por slot seja atendida.	Depende de outras restrições, como pausas e limites diários, que podem reduzir a alocação.
    # Limite Diário	Limita a condução a 36 períodos (9h) por dia.	Entra em conflito com a necessidade de atender a demanda alta em dias específicos.
    # Pausa de 45 Minutos	Requer uma pausa após 18 períodos (4.5h) de condução.	Pode reduzir a capacidade de atender à demanda contínua, especialmente em slots críticos.
    # Pausa Fracionada 15+30	Divide a pausa de 45 minutos em 15 e 30 minutos.	Complementa a pausa de 45 minutos, mas adiciona complexidade na alocação.
    # Pausa Fracionada 30+15	Alternativa à pausa de 45 minutos, começando com 30 minutos.	Mesmo impacto da pausa 15+30, mas requer flexibilidade na sequência de slots.
    # Repouso Diário Mínimo	Impõe um repouso de 11h (44 períodos) por dia.	Conflita com demandas altas em períodos consecutivos.
    # Repouso Diário Reduzido	Permite repouso de 9h (36 períodos), mas limitado a 3 dias em 14 dias.	Pode aliviar o impacto do repouso diário mínimo, mas aumenta a complexidade de planejamento.
    # Limite Semanal	Limita a 224 períodos (56h) por semana.	Restrições diárias e pausas impactam a soma semanal, reduzindo a flexibilidade.
    # Limite Quinzenal	Limita a 360 períodos (90h) por duas semanas.	Conflita com demandas contínuas em períodos de alta demanda.
    # Repouso Semanal	Impõe repouso de 45h (180 períodos) por semana.	Impacta diretamente a cobertura de necessidade semanal.
    # Repouso Quinzenal	Exige repouso de 24h (96 períodos) a cada duas semanas.	Menor impacto direto, mas afeta planejamento a longo prazo.
    # Descanso Após Trabalho	Requer descanso após 6 dias consecutivos de trabalho.	Complementa os limites semanais e quinzenais, mas pode criar conflitos em semanas críticas.

    # # Prioridade das Restrições
    # prioridades = [
    #     "cobertura_necessidade",  # Garantir que a demanda seja atendida
    #     "pausa_45_minutos",       # Garantir pausas obrigatórias
    #     "limite_diario",          # Limitar as horas diárias de condução
    #     "repouso_diario_minimo",  # Respeitar repouso diário mínimo
    #     "limite_quinzenal",       # Limitar o total de horas quinzenais
    # ]
    
    constraints = []
    # msg_result = []
    num_periods = len(need)
    Y = {}

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    # solver.EnableOutput()

    # if radio_selection_object is None:
    #     raise ValueError("radio_selection_object must be provided")

    # radio_selection_object só é obrigatório no modo EXATO
    if mode == "Exact" and radio_selection_object is None:
        raise ValueError("radio_selection_object must be provided for Exact mode")


    
    # Parâmetros de controle
    # max_demands_per_driver = 3  # Cada motorista pode atender até 3 demandas por período
    # penalty = 0.01  # Penalidade reduzida para demanda não atendida
    # max_total_motoristas = 500  # Total de motoristas disponíveis no modelo
    # tolerancia_demanda = 0.9  # Aceitar 90% da demanda como suficiente
    
    # Alocação inicial utilizando a heurística gulosa
    # allocation = greedy_allocation(need, limit_Workers, num_periods, limit_Workers,max_demands_per_driver)
    # allocation = None
    with contextlib.redirect_stdout(stdout_buffer):
        with contextlib.redirect_stderr(stderr_buffer):    
    
            # Alocação inicial (pode vir da heurística externa ou ser None)
            allocation = initial_allocation

            
            # Criar o solver
            solver = pywraplp.Solver.CreateSolver(solver_param_type)  # Usar 'GLOP' para LP ou 'SCIP' para MIP
            solver.EnableOutput()


            # st.markdown(f"**Mode:** {optimization_mode} &nbsp;&nbsp;|&nbsp;&nbsp; **Status:** {status} &nbsp;&nbsp;|&nbsp;&nbsp; **Total drivers:** {total_workers}")

            for d in range(num_periods):
                for t in range(limit_workers):
                    if variable_type == "Continuous":
                        Y[d, t] = solver.NumVar(0, solver.infinity(), f'Y[{d}, {t}]')  # Variável contínua
                    elif variable_type == "Binary":
                        Y[d, t] = solver.BoolVar(f'Y[{d}, {t}]')  # Variável binária
                    elif variable_type == "Integer":
                        Y[d, t] = solver.IntVar(0, solver.infinity(), f'Y[{d}, {t}]')  # Variável inteira
                    else:
                        raise ValueError("Invalid variable type for Y[d, t].")


            # # Aplicar alocação inicial com a heurística (forçar valores de Y com base na alocação gulosa)
            # for d in range(num_periods):
            #     for t in range(limit_Workers):
            #         if allocation[d, t] == 1:
            #             solver.Add(Y[d, t] == 1)  # Força a alocação do motorista para o período



            if (radio_selection_object == "Maximize Demand Response"):
                # Função objetivo: minimizar o número total de trabalhadores
                # Função objetivo: maximizar o atendimento de demanda considerando as tolerâncias e limites
                solver.Maximize(
                    solver.Sum(Y[d, t] * need[d] for d in range(num_periods) for t in range(limit_workers))  # Maximizar o atendimento da demanda
                    - penalty * solver.Sum(
                        (need[d] * tolerance_demands - solver.Sum(Y[d, t] * max_demands_per_driver for t in range(limit_workers)))
                        for d in range(num_periods)  # Penalização para garantir que a demanda atendida seja dentro da tolerância
                    )
                )

                # Restrição para distribuir o atendimento de forma proporcional:    
                for d in range(num_periods):
                    solver.Add(solver.Sum(Y[d, t] for t in range(limit_workers)) <= need[d])  # Garantir que a demanda não seja superada por período

            elif (radio_selection_object == "Minimize Total Number of Drivers"):
                # Função objetivo ajustada
                solver.Minimize(
                    solver.Sum(Y[d, t] for d in range(num_periods) for t in range(limit_workers))  # Minimizar motoristas
                    + penalty * solver.Sum((need[d] * tolerance_demands - solver.Sum(Y[d, t] * max_demands_per_driver for t in range(limit_workers))) for d in range(num_periods))
                )
            # else:
            #     raise ValueError(f"Invalid radio_selection_object: {radio_selection_object}")                
            
            #01
            if selected_restrictions["cobertura_necessidade"]: 
                for d in range(num_periods):
                    constraint_expr = solver.Sum(
                        Y[d, t] * max_demands_per_driver for t in range(limit_workers)) >= tolerance_demands * need[d]  # A demanda deve ser atendida

                    # Adiciona a restrição garantindo que a soma seja maior ou igual à necessidade para o período
                    constraint = solver.Add(constraint_expr)
                    constraints.append(constraint)
        
            # 1) Fixar parte da solução (LNS): tudo que vem em fixed_assignments é imposto
            # if fixed_assignments is not None:
            #     for (d_fix, t_fix, val) in fixed_assignments:
            #         # Garante que Y[d_fix, t_fix] assuma o valor da solução atual fora da vizinhança
            #         solver.Add(Y[d_fix, t_fix] == val)
                    
            if fixed_assignments:
                for (d_fix, t_fix, val) in fixed_assignments:
                    # SCIP requer variáveis booleanas explicitamente fixadas assim:
                    if val == 1:
                        Y[d_fix, t_fix].SetBounds(1, 1)
                    else:
                        Y[d_fix, t_fix].SetBounds(0, 0)                    

            # 2) Sugerir uma solução inicial (heurística) SEM forçar tudo
            # if allocation is not None:
            #     for d in range(num_periods):
            #         for t in range(limit_workers):
            #             if allocation[d, t] == 1:
            #                 # Em vez de fixar igual a 1, você pode só "incentivar" via callback de warm-start
            #                 # ou, numa versão simples, deixar comentado até estudar melhor:
            #                 # solver.Add(Y[d, t] >= 0.5)
            #                 pass

            #02
            if selected_restrictions["pausa_45_minutos"]: 
                max_continuous_work = 18  # 4,5 horas de trabalho contínuo (18 períodos de 15 minutos)
                pause_duration = 3  # 45 minutos de pausa (3 períodos de 15 minutos)
                max_daily_driving = 36  # 9 horas de trabalho diário (36 períodos de 15 minutos)

                for t in range(limit_workers):  # Para cada motorista
                    for start in range(num_periods - max_continuous_work - pause_duration):
                        # Soma dos períodos de trabalho nos próximos 18 slots (4,5 horas)
                        continuous_work = solver.Sum(Y[start + p, t] for p in range(max_continuous_work))
                        
                        # Soma dos períodos de pausa nos próximos 3 slots
                        pause = solver.Sum(Y[start + max_continuous_work + p, t] for p in range(pause_duration))
                        
                        # Garantir que após 18 períodos de trabalho, haja uma pausa de 45 minutos
                        constraint_work_pause = solver.Add(continuous_work <= max_continuous_work - pause)
                        constraints.append(constraint_work_pause)

                    # Garantir que o motorista não ultrapasse o limite diário de 9 horas de condução
                    # daily_driving = solver.Sum(Y[d, t] for d in range(num_periods))
                    # constraint_daily_driving = solver.Add(daily_driving <= max_daily_driving)
                    # constraints.append(constraint_daily_driving)
            
            #03
            if selected_restrictions["limite_diario"]: 
                max_daily_driving = 36  # 9 horas (36 períodos de 15 minutos)
                for day in range(num_periods // 96):  # 96 períodos de 15 minutos em um dia
                    
                    # Cria a expressão para a soma de X[j] nos períodos válidos
                    day_start = day * 96  # Início do dia em termos de períodos
                    day_end = (day + 1) * 96  # Fim do dia em termos de períodos (não inclusivo)
                    
                    # Expressão para a soma dos períodos de trabalho (condução ou pausas)
                    constraint_expr = solver.Sum(
                        Y[d, t] for d in range(day_start, day_end)
                    )
                    
                    # Ajuste do limite diário de acordo com a necessidade de repouso reduzido
                    # Se o repouso for reduzido, podemos permitir mais períodos de trabalho
                    max_daily_with_rest = max_daily_driving
                    
                    # Adiciona a restrição de limite diário, mas com mais flexibilidade se o repouso for reduzido
                    constraint = solver.Add(constraint_expr <= max_daily_with_rest)
                    constraints.append(constraint)
        
            #4
            if selected_restrictions["repouso_diario_minimo"]:    
                daily_rest_periods = 44  # 11 horas = 44 períodos de 15 minutos
                reduced_rest_periods = 36  # 9 horas = 36 períodos de 15 minutos
                reduced_rest_days = 3  # Número de dias que pode reduzir o repouso para 9 horas (36 períodos)
                
                reduced_rest_vars = []  # Lista para armazenar as variáveis de repouso reduzido
                
                for day in range(num_periods // 96):  # Para cada dia com 96 períodos de 15 minutos
                    reduced_rest = solver.BoolVar(f'reduced_rest[{day}]')  # Variável booleana para indicar repouso reduzido
                    reduced_rest_vars.append(reduced_rest)
                    
                    # Restrição de repouso: 44 períodos ou 36 períodos se o repouso for reduzido
                    # A ideia é permitir mais flexibilidade nos dias com repouso reduzido
                    constraint = solver.Add(
                        solver.Sum(
                            Y[day * 96 + p, t] for p in range(96) for t in range(limit_workers)  # Para cada motorista t
                        ) <= (96 - reduced_rest_periods) + reduced_rest_periods * reduced_rest  # Ajuste dependendo se o repouso foi reduzido
                    )
                    constraints.append(constraint)
                
                # Adiciona a restrição que permite no máximo 3 dias com repouso reduzido
                constraint = solver.Add(solver.Sum(reduced_rest_vars) <= reduced_rest_days)
                constraints.append(constraint)

            #5
            if selected_restrictions["limite_quinzenal"]:    
                # Para cada período de 14 dias, com 96 períodos por dia
                for period in range(num_periods // (96 * 14)):  # 96 períodos/dia * 14 dias
                    # Expressão para somar os períodos válidos de condução e pausa
                    constraint_expr = solver.Sum(
                        # X[period * 96 * 14 + p] for p in range(96 * 14)
                        Y[period * 96 + p, t] for p in range(96 * 14) for t in range(limit_workers)  # Para cada motorista t
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

            #6
            if selected_restrictions["repouso_diario_reduzido"]:    
                reduced_weekly_rest_weeks = 1  # Quantas vezes pode reduzir o repouso para 24 horas a cada duas semanas
                weekly_rest_vars = []  # Lista para armazenar as variáveis de repouso semanal reduzido

                for week in range(num_periods // (96 * 7)):  # Para cada semana com 96 * 7 períodos de 15 minutos
                    reduced_rest = solver.BoolVar(f'reduced_weekly_rest[{week}]')  # Variável booleana para indicar repouso semanal reduzido
                    weekly_rest_vars.append(reduced_rest)
                    
                    # Restrição para o repouso semanal: 
                    constraint = solver.Add(
                        solver.Sum(
                            # X[week * 96 * 7:(week + 1) * 96 * 7]
                            Y[week * 96 * 7:(week + 1) * 96 * 7, t] for t in range(limit_workers)  # Para cada motorista t

                            ) >= 180 * (1 - reduced_rest) + 96 * reduced_rest
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
                        # X[day * 96 + p] for p in range(96)
                        Y[day * 96 + p, t] for p in range(96) for t in range(limit_workers)  # Para cada motorista t
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
                # weekly_periods = solver.Sum(
                    # X[i, j] for i in range(num_periods) for j in range(num_periods))

                # Soma total de períodos na semana, com a possibilidade de dias com 10 horas
                weekly_periods = solver.Sum(
                    Y[d, t] for d in range(num_periods) for t in range(limit_workers)  # Soma de todos os períodos e motoristas
                )


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
                            constraint = solver.Add(solver.Sum(
                                # X[i, j] for i in range(i, j)
                                Y[i, t] for t in range(limit_workers)
                                ) <= 0)  # Impede o trabalho sem descanso após 6 dias consecutivos

                            # Esta lógica pode ser personalizada dependendo da forma como os períodos de descanso são distribuídos
                            constraints.append(constraint)

            # # Se "divisao_pausa" também estiver marcado, então aplica a lógica de divisão da pausa
            if selected_restrictions.get("divisao_pausa1530", False):
                for start in range(num_periods - 19):  # 18 períodos = 4,5 horas de condução
                    # Pausa de 15 minutos seguida de 30 minutos
                    first_half_pause = solver.Sum(Y[start + 18, t] for t in range(limit_workers))
                    second_half_pause = solver.Sum(Y[start + 19, t] for t in range(limit_workers))
                    
                    solver.Add(first_half_pause >= 1)  # Pausa de 15 minutos
                    solver.Add(second_half_pause >= 1)  # Pausa de 30 minutos
                    constraints.append(solver)
                    
            # # Se "divisao_pausa" também estiver marcado, então aplica a lógica de divisão da pausa
            if selected_restrictions.get("divisao_pausa3015", False):
                # Lógica para dividir a pausa de 45 minutos em 15 e 30 minutos, aplicando a verificação do fracionamento
                for start in range(num_periods - 19):  # 18 períodos = 4,5 horas de condução
                    # Pausa de 15 minutos seguida de 30 minutos
                    first_half_pause = solver.Sum(Y[start + 19, t] for t in range(limit_workers))
                    second_half_pause = solver.Sum(Y[start + 18, t] for t in range(limit_workers))
                    solver.Add(first_half_pause >= 1)  # Pausa de 15 minutos
                    solver.Add(second_half_pause >= 1)  # Pausa de 30 minutos
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

            # # A multiplicação de uma linha por uma constante é útil se for usada para cancelar ou simplificar valores, mas pode aumentar o fill-in se não for cuidadosamente controlada.
            # if multiply_row is not None:
            #     # Exibir o sistema linear antes da troca de linhas
            #     row, constant = multiply_row
            #     if 0 <= row < len(constraints_coefficients):
            #         if constant != 0:
            #             # Multiplicação dos coeficientes por uma constante
            #             constraints_coefficients[row] = [c * constant for c in constraints_coefficients[row]]

            #             # Atualização da restrição correspondente no solver
            #             new_expr = solver.Sum(
            #                 # constraints_coefficients[row][j] * X[j] for j in range(len(X))
            #                 constraints_coefficients[row][d][t] * Y[d, t]  for d in range(num_periods) for t in range(limit_Workers)
            #                 )
            #             constraints[row] = solver.Add(new_expr >= constraints[row].lb() * constant)
            #             print(f"Row {row} multiplied by {constant} successfully!")

            # A multiplicação de uma linha por uma constante pode ser útil para simplificar valores,
            # mas deve ser controlada para evitar o aumento do "fill-in".
            if multiply_row is not None:
                row, constant = multiply_row
                
                # Verificar se o índice da linha está dentro dos limites da matriz
                if 0 <= row < len(constraints_coefficients):
                    
                    # Evitar multiplicação por 0
                    if constant != 0:
                        
                        # Multiplicar a linha pelos coeficientes, levando em consideração a estrutura de 3D
                        for d in range(len(constraints_coefficients[row])):
                            if isinstance(constraints_coefficients[row][d], (list, np.ndarray)):  # Verifica se é uma estrutura iterável
                                # Se for uma lista ou array, iterar sobre t
                                for t in range(len(constraints_coefficients[row][d])):
                                    constraints_coefficients[row][d][t] *= constant
                            else:
                                # Se for um valor escalar, apenas multiplica diretamente
                                constraints_coefficients[row][d] *= constant

                        # Atualizar a expressão de restrição correspondente no solver
                        new_expr = solver.Sum(
                            # Agora verificamos se constraints_coefficients[row][d] é uma estrutura iterável
                            (constraints_coefficients[row][d][t] if isinstance(constraints_coefficients[row][d], (list, np.ndarray)) 
                            else constraints_coefficients[row][d]) * Y[d, t]
                            for d in range(num_periods)
                            for t in range(limit_workers)
                        )
                        
                        # Atualizar a restrição com a multiplicação da constante
                        constraints[row] = solver.Add(new_expr >= constraints[row].lb() * constant)
                        
                        # Mensagem de sucesso
                        print(f"Row {row} multiplied by {constant} successfully!")
                    else:
                        print(f"Skipping multiplication for row {row} as constant is zero.")
                else:
                    print(f"Invalid row index {row} for constraints_coefficients.")

            # #Multiplicacao automatica
            # if add_multiple_rows is not None:
            #     # display_system(constraints_coefficients, need)
            #     row1, row2, multiple = add_multiple_rows
            #     if 0 <= row1 < len(constraints_coefficients) and 0 <= row2 < len(constraints_coefficients):
            #         # Adicionar múltiplo da linha row1 à linha row2
            #         # new_row_values = [
            #         #     constraints_coefficients[row2][j] + multiple * constraints_coefficients[row1][j] for j in range(len(X))
            #         # ]
                
            #         # Adicionar múltiplo da linha row1 à linha row2
            #         new_row_values = [
            #             [ constraints_coefficients[row2][d][t] + multiple * constraints_coefficients[row1][d][t] for t in range(limit_Workers) ]
            #             for d in range(num_periods)
            #         ]
                
                
            #         # Verificar se a linha resultante tem valores não nulos
            #         if any(value != 0 for value in new_row_values):
            #             # Atualizar a linha row2 com os novos valores calculados
            #             constraints_coefficients[row2] = new_row_values

            #             # Atualizar o lado direito da equação para a linha row2 no vetor need
            #             need[row2] = need[row2] + multiple * need[row1]

            #             # Multiplicar a linha por -1 se houver coeficientes negativos
            #             if any(value < 0 for value in constraints_coefficients[row2]):
            #                 constraints_coefficients[row2] = [-value for value in constraints_coefficients[row2]]
            #                 need[row2] = -need[row2]

            #             # Atualizar a restrição no solver com os novos valores
            #             new_expr = solver.Sum(
            #                 # constraints_coefficients[row2][j] * X[j] for j in range(len(X))
            #                 constraints_coefficients[row2][d][t] * Y[d, t] for d in range(num_periods) for t in range(limit_Workers)
            #                 )
            #             constraints[row2] = solver.Add(new_expr >= need[row2])

            #         else:
            #             st.write("The operation resulted in a null line, which was avoided.")

            # # Multiplicação automática de linhas
            # if add_multiple_rows is not None:
            #     row1, row2, multiple = add_multiple_rows
                
            #     # Verifique se os índices das linhas são válidos
            #     if 0 <= row1 < len(constraints_coefficients) and 0 <= row2 < len(constraints_coefficients):
                    
            #         # Verifique se constraints_coefficients[row1] e constraints_coefficients[row2] são arrays 2D
            #         if isinstance(constraints_coefficients[row1], np.ndarray) and constraints_coefficients[row1].ndim == 2:
            #             print(f"Row {row1} is 2D array with shape: {constraints_coefficients[row1].shape}")
            #         else:
            #             print(f"Warning: Row {row1} is not a 2D array. Type: {type(constraints_coefficients[row1])}")

            #         if isinstance(constraints_coefficients[row2], np.ndarray) and constraints_coefficients[row2].ndim == 2:
            #             print(f"Row {row2} is 2D array with shape: {constraints_coefficients[row2].shape}")
            #         else:
            #             print(f"Warning: Row {row2} is not a 2D array. Type: {type(constraints_coefficients[row2])}")

            #         # Se ambos forem arrays 2D, proceda com a operação de adição
            #         if isinstance(constraints_coefficients[row1], np.ndarray) and constraints_coefficients[row1].ndim == 2 and \
            #         isinstance(constraints_coefficients[row2], np.ndarray) and constraints_coefficients[row2].ndim == 2:
                        
            #             # Criar os novos valores para a linha row2, adicionando o múltiplo de row1
            #             new_row_values = [
            #                 [
            #                     (constraints_coefficients[row2][d][t] + multiple * constraints_coefficients[row1][d][t])
            #                     for t in range(limit_Workers)
            #                 ]
            #                 for d in range(num_periods)
            #             ]
                        
            #             # Verificar se a linha resultante tem valores não nulos
            #             if any(value != 0 for row in new_row_values for value in row):
                            
            #                 # Atualizar constraints_coefficients[row2] com os novos valores
            #                 constraints_coefficients[row2] = np.array(new_row_values)

            #                 # Atualizar o lado direito da equação para a linha row2 no vetor 'need'
            #                 need[row2] += multiple * need[row1]

            #                 # Multiplicar a linha por -1 se houver coeficientes negativos
            #                 if any(value < 0 for value in constraints_coefficients[row2]):
            #                     constraints_coefficients[row2] = [-value for value in constraints_coefficients[row2]]
            #                     need[row2] = -need[row2]

            #                 # Atualizar a restrição no solver com os novos valores
            #                 new_expr = solver.Sum(
            #                     constraints_coefficients[row2][d][t] * Y[d, t]
            #                     for d in range(num_periods)
            #                     for t in range(limit_Workers)
            #                 )
            #                 constraints[row2] = solver.Add(new_expr >= need[row2])

            #                 print(f"Row {row2} updated successfully.")
            #             else:
            #                 print("The operation resulted in a null line, which was avoided.")
            # Multiplicação automática de linhas
            # Multiplicação automática de linhas
            if add_multiple_rows is not None:
                row1, row2, multiple = add_multiple_rows

                # Exibir informações de depuração sobre a estrutura da matriz
                print(f"Structure of constraints_coefficients: {type(constraints_coefficients)}, Shape: {np.array(constraints_coefficients).shape}")

                # Verifique se os índices das linhas são válidos
                if 0 <= row1 < len(constraints_coefficients) and 0 <= row2 < len(constraints_coefficients):
                    print(f"Applying add_multiple_rows operation: row1={row1}, row2={row2}, multiple={multiple}")

                    # Verificar se constraints_coefficients[row1] e constraints_coefficients[row2] são listas ou arrays
                    if isinstance(constraints_coefficients[row1], (list, np.ndarray)) and isinstance(constraints_coefficients[row2], (list, np.ndarray)):
                        
                        # Criar os novos valores para a linha row2, adicionando o múltiplo de row1
                        new_row_values = [
                            constraints_coefficients[row2][j] + multiple * constraints_coefficients[row1][j]
                            for j in range(len(constraints_coefficients[row2]))
                        ]

                        # Verificar se a linha resultante tem valores não nulos (com tolerância para zero)
                        if any(abs(value) > 1e-6 for value in new_row_values):
                            constraints_coefficients[row2] = new_row_values
                            need[row2] = need[row2] + multiple * need[row1]

                            # Multiplicar a linha por -1 se houver coeficientes negativos
                            if any(value < 0 for value in constraints_coefficients[row2]):
                                constraints_coefficients[row2] = [-value for value in constraints_coefficients[row2]]
                                need[row2] = -need[row2]

                            # Atualizar a restrição no solver com os novos valores
                            try:
                                new_expr = solver.Sum(
                                    constraints_coefficients[row2][j] * Y[d, t]
                                    for j in range(len(constraints_coefficients[row2]))
                                    for d in range(num_periods)
                                    for t in range(limit_workers)
                                    if (d, t) in Y  # Verificar se a chave existe no dicionário
                                )
                                constraints[row2] = solver.Add(new_expr >= need[row2])
                                print(f"Row {row2} updated successfully: {constraints_coefficients[row2]}")
                            except KeyError as e:
                                print(f"KeyError: {e}. Invalid key in Y or mismatch in constraints_coefficients.")
                            except Exception as e:
                                print(f"Unexpected error: {e}")
                        else:
                            print(f"The operation resulted in a null line for row {row2}, which was avoided.")
                    else:
                        print(f"Error: Unexpected structure for rows {row1} or {row2}. Ensure constraints_coefficients is 2D.")
                else:
                    print(f"Invalid indices for add_multiple_rows operation: row1={row1}, row2={row2}")

            msgResult = None        

            
            # with contextlib.redirect_stdout(log_buffer):
            #     status = solver.Solve()

            # solver_logs = log_buffer.getvalue()
            
            # with st.expander("Solver Logs", expanded=True):
            #     st.text_area("Logs", solver_logs, height=300)


            # solver.wall_time()
            # solver.nodes()
            # solver.iterations()
            # solver.Objective().Value()

            # st.write(f"Wall time: {solver.wall_time()} ms")
            # st.write(f"Iterations: {solver.iterations()}")
            # st.write(f"Branches: {solver.nodes()}")


            # Dentro da sua função solve_shift_schedule
            final_density = calculate_density(constraints_coefficients)
            # Calcular densidade
            initial_density = calculate_density(initial_constraints_coefficients)
            
            status = None
            total_workers = 0
            workers_schedule = 0
            
            # Inicializando uma lista para armazenar os resultados
            statistics_result = []
            iterations_data = []
            initial_relaxation_rate = 0.01
            max_penalty = 10
            
            # Apenas registra a densidade, mas NÃO bloqueia o Solve
            if densidade_aceitavel is not None:
                statistics_result.append(
                    f"Density check: final={final_density:.4f}, threshold={densidade_aceitavel:.4f}"
                )
            
            
            # if final_density <= densidade_aceitavel:
            # Resolver o problema
            status = solver.Solve()
            
            # Iterar para resolver conflitos
            max_iterations = limit_iteration
            relaxation_level = limit_level_relaxation  # Relaxa as restrições progressivamente
            iteration = 0
            
            while status != pywraplp.Solver.OPTIMAL and iteration < max_iterations:
                
                # Relaxar a penalização e as restrições conforme a iteração
                relaxation_level = iteration * initial_relaxation_rate  # O nível de relaxamento aumenta com o tempo
                
                # Ajustar a função objetivo com a penalização relaxada
                objective = relax_objective(solver, Y, num_periods, limit_workers, penalty, max_penalty, relaxation_level, need, tolerance_demands, max_demands_per_driver)

                # Capturar dados da iteração
                iteration_data = {
                    "iteration": iteration,
                    "relaxation_level": relaxation_level,
                    "status": get_solver_status_description(status),
                    "objective_value": solver.Objective().Value() if status in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE} else 0,
                }
                
                

                # Relaxar restrições relacionadas
                if selected_restrictions["limite_diario"]:
                    relax_restrictions(solver, constraints, relaxation_level)
                if selected_restrictions["limite_semanal"]:
                    relax_restrictions(solver, constraints, relaxation_level)
                if selected_restrictions["repouso_diario_minimo"]:
                    relax_restrictions(solver, constraints, relaxation_level)
        
                # Resolver novamente
                status = solver.Solve()

                iterations_data.append(iteration_data)
                
                iteration += 1



            if status == pywraplp.Solver.OPTIMAL:
                
                # Parte para calcular o total de motoristas por período (sem erros de iteração)
                # workers_schedule = [sum(int(Y[d, t].solution_value()) for d in range(num_periods)) for t in range(limit_Workers)]

                # # Parte da penalização
                # penalty_sum = solver.Sum(
                #     (need[d] * tolerance_Demands - solver.Sum(Y[d, t] * max_demands_per_driver for t in range(limit_Workers)))
                #     for d in range(num_periods)
                # )

                # # A função objetivo deve ser ajustada com a penalização corretamente
                # solver.Minimize(
                #     solver.Sum(Y[d, t] for d in range(num_periods) for t in range(limit_Workers))  # Minimizar motoristas
                #     + penalty * penalty_sum  # Penalização ajustada
                # )
                
                
                # workers_schedule = [sum(int(Y[d, t].solution_value()) for d in range(num_periods)) for t in range(limit_Workers) + penalty * solver.Sum((need[d] * tolerance_Demands - solver.Sum(Y[d, t] * max_demands_per_driver for t in range(limit_Workers))) for d in range(num_periods))]
                # workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
                workers_schedule = [sum(int(Y[d, t].solution_value()) for t in range(limit_workers)) for d in range(num_periods)]
                total_workers = solver.Objective().Value()
                statistics_result.append(f"Model State: OPTIMAL")

            elif status == pywraplp.Solver.FEASIBLE:
                statistics_result.append(f"Model State: FEASIBLE")
            elif status == pywraplp.Solver.INFEASIBLE:
                statistics_result.append(f"Model State: INFEASIBLE")
            elif status == pywraplp.Solver.UNBOUNDED:
                statistics_result.append(f"Model State: UNBOUNDED")
            elif status == pywraplp.Solver.ABNORMAL:
                statistics_result.append(f"Model State: ABNORMAL")
            elif status == pywraplp.Solver.MODEL_INVALID:
                statistics_result.append(f"Model State: MODEL_INVALID")
            elif status == pywraplp.Solver.NOT_SOLVED:
                statistics_result.append(f"Model State: NOT_SOLVED")
            else:
                statistics_result.append(f"Model State: NOT_SOLVED")
                
                # Função objetivo ajustada
                # solver.Minimize(
                #     solver.Sum(Y[d, t] for d in range(num_periods) for t in range(limit_Workers))  # Minimizar motoristas
                #     + penalty * solver.Sum((need[d] * tolerance_Demands - solver.Sum(Y[d, t] * max_demands_per_driver for t in range(limit_Workers))) for d in range(num_periods))
                # )
                
                # Soma as alocações de todos os motoristas (t) para cada período (d)
                # workers_schedule = [sum(int(Y[d, t].solution_value()) for t in range(limit_Workers)) for d in range(num_periods)]
                workers_schedule = [sum(int(Y[d, t].solution_value()) for d in range(num_periods)) for t in range(limit_workers) + penalty * solver.Sum((need[d] * tolerance_demands - solver.Sum(Y[d, t] * max_demands_per_driver for t in range(limit_workers))) for d in range(num_periods))]
                # workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
                total_workers = solver.Objective().Value()            

            # Adicionando o tipo do modelo aos resultados
            statistics_result.append(f"Total workers needed: {total_workers}")

            # Adicionando o tipo do modelo aos resultados
            statistics_result.append(f"Model Type: {tipo_modelo(solver)}")

            # Adicionando estatísticas do solver aos resultados
            statistics_result.append(f"Total Resolution Time: {solver.wall_time()} ms")
            statistics_result.append(f"Total Number of Iterations: {solver.iterations()}")
            statistics_result.append(f"Number of Restrictions: {solver.NumConstraints()}")
            statistics_result.append(f"Number of Variables: {solver.NumVariables()}")
            # else:
            #     statistics_result.append(f"Model Status: Density not acceptable")
            #     # workers_schedule = [int(X[d].solution_value()) for d in range(num_periods)]
            #     workers_schedule = [sum(int(Y[d, t].solution_value()) for t in range(limit_workers)) for d in range(num_periods)]
            #     total_workers = solver.Objective().Value()

            #     # Adicionando o tipo do modelo aos resultados
            #     statistics_result.append(f"Total workers Needed: {total_workers}")

            #     # Adicionando o tipo do modelo aos resultados
            #     statistics_result.append(f"Model Type: {tipo_modelo(solver)}")

            #     # Adicionando estatísticas do solver aos resultados
            #     statistics_result.append(f"Total Resolution Time: {solver.wall_time()} ms")
            #     statistics_result.append(f"Total Number of Iterations: {solver.iterations()}")
            #     statistics_result.append(f"Number of Restrictions: {solver.NumConstraints()}")
            #     statistics_result.append(f"Number of Variables: {solver.NumVariables()}")
                
            save_data(constraints_coefficients, 'constraints_coefficients.json')
            
    # Captura final dos logs
    solver_stdout = stdout_buffer.getvalue() if stdout_buffer else ""
    solver_stderr = stderr_buffer.getvalue() if stderr_buffer else ""

    # Anexa os logs no resultado
    solver_logs = {
        "stdout": solver_stdout,
        "stderr": solver_stderr
    }            

    # st.write("DEBUG solver_logs type:", type(solver_logs))
    # st.write("DEBUG solver_logs:", solver_logs)

    return solver, status, total_workers, workers_schedule, constraints_coefficients, initial_density, final_density, statistics_result, msg_result, iterations_data, allocation, solver_logs

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
def preencher_restricoes(initial_constraints_coefficients, restrictions, selected_restrictions, num_periods, need):
    # Loop para preencher a matriz de restrições
    for i in range(num_periods):
        for j in range(num_periods):
            # Calcula a diferença cíclica (i - j) com modularidade
            diff = (i - j) % num_periods
            
            # Inicializa como sem cobertura
            initial_constraints_coefficients[i, j] = 0
            
            # Itera sobre as restrições
            for restriction in restrictions:
                # Verifica se a restrição está selecionada
                if selected_restrictions.get(restriction["Key"], False):
                    
                    # Aplique a lógica para cada restrição
                    if restriction["Key"] == "limite_diario":

                        # Primeira janela: condução inicial (0 a 18 períodos)
                        if 0 <= (i - j) % num_periods < 18:
                            initial_constraints_coefficients[i, j] = 1  # Condução permitida

                        # Segunda janela: pausa obrigatória (19 a 20 períodos) Pausa de 15 minutos
                        elif 19 <= (i - j) % num_periods < 25 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 25 <= (i - j) % num_periods < 28 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 30 <= (i - j) % num_periods < 37 and selected_restrictions.get("divisao_pausa1530", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        # Terceira janela: pausa obrigatória (19 a 20 períodos) Pausa de 30 minutos
                        elif 20 <= (i - j) % num_periods < 21 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 30 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 21 <= (i - j) % num_periods < 25 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 26 <= (i - j) % num_periods < 37 and selected_restrictions.get("divisao_pausa3015", False):  # Pausa de 15 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        # Quarta janela: pausa obrigatória (19 a 20 períodos) Pausa de 30 minutos
                        elif 21 <= (i - j) % num_periods < 37 and selected_restrictions.get("pausa_45_minutos", False):  # Pausa de 30 minutos ocupa 1 período
                            initial_constraints_coefficients[i, j] = 1  # Pausa fracionada permitida
                        elif 0 <= (i - j) % num_periods < 44 and selected_restrictions.get("repouso_diario_minimo", False):
                            # Exemplo de condição para repouso diário mínimo
                            #if 0 <= diff < 44:  # Repouso diário de 11h
                            initial_constraints_coefficients[i, j] = 1    
                        elif 0 <= (i - j) % num_periods < 36 and selected_restrictions.get("repouso_diario_reduzido", False): #restriction["Key"] == "repouso_diario_reduzido":
                            # Exemplo de condição para repouso diário reduzido
                                # if 0 <= diff < 36:  # Repouso de 9h
                            initial_constraints_coefficients[i, j] = 1                                                    
                        # Fora dos intervalos permitidos
                        else:
                            initial_constraints_coefficients[i, j] = 0  # Fora do intervalo permitido
                    elif restriction["Key"] == "repouso_semanal":
                        # Exemplo de condição para repouso semanal
                        if 0 <= diff < 180:  # Repouso semanal de 45h
                            initial_constraints_coefficients[i, j] = 1
                    elif restriction["Key"] == "repouso_quinzenal":
                        # Exemplo de condição para repouso quinzenal
                        if 0 <= diff < 96:  # Repouso quinzenal de 24h
                            initial_constraints_coefficients[i, j] = 1
                    elif restriction["Key"] == "descanso_apos_trabalho":
                        # Exemplo de condição para descanso após dias de trabalho
                        if 0 <= diff < 6:  # Após 6 dias de trabalho, descanso
                            initial_constraints_coefficients[i, j] = 1
                    elif restriction["Key"] == "limite_semanal":
                        # Limite semanal de condução (224 períodos)
                        # Verifica a soma de períodos dentro de uma semana
                        week_start = (i // 7) * 7
                        week_end = week_start + 7
                        weekly_periods = np.sum(initial_constraints_coefficients[week_start:week_end, :])
                        if weekly_periods <= 224:
                            initial_constraints_coefficients[i, j] = 1
                    elif restriction["Key"] == "limite_quinzenal":
                        # Limite quinzenal de condução (360 períodos)
                        # Verifica a soma de períodos dentro de uma quinzena
                        fortnight_start = (i // 14) * 14
                        fortnight_end = fortnight_start + 14
                        fortnight_periods = np.sum(initial_constraints_coefficients[fortnight_start:fortnight_end, :])
                        if fortnight_periods <= 360:
                            initial_constraints_coefficients[i, j] = 1

                    elif restriction["Key"] == "cobertura_necessidade":
                        # Cobertura de necessidade (verifica a necessidade de trabalhadores)
                        if np.sum(initial_constraints_coefficients[:, i]) >= need[i]:
                            initial_constraints_coefficients[i, j] = 1

                            
    return initial_constraints_coefficients

# Interface do Streamlit
st.title("Shift Scheduler")

# Inicializa a variável default_need vazia
default_need = []
need_input = None
num_periods = None

initial_constraints_coefficients = []

# Carrega os dados do arquivo, se existirem
initial_constraints_coefficients = load_data('initial_constraints_coefficients.json')


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
            tolerance_demands = st.number_input("Tolerance Coverage Demands", min_value=0.01, value=0.01)
            penalty = st.number_input("Penalty for unmet demand", min_value=0.01, value=0.01)

        with col2:
            st.write("Algorithm")
            variable_type = st.selectbox("Variable", ["Integer", "Binary", "Continuous"])
            solver_param_type = st.selectbox("GLOP-LP | SCIP-MIP", ["SCIP", "GLOP"])
            acceptable_percentage = st.number_input("Acceptable Density", min_value=0.01)
            
        with col3:
            st.write("Iterations|Relaxation")
            limit_iteration = st.number_input("Limit Iterations", min_value=0, value=0)
            limit_level_relaxation = st.number_input("Relaxation", min_value=0, value=0)
        with col4:
            fixar_valores = st.checkbox("Set Values", value=True)

            # Condicional para definir `default_need` com valores fixos ou novos aleatórios
            if fixar_valores:
                default_need = gerar_valores_aleatorios(total_hours, period_minutes)
            else:
                default_need = np.random.randint(1, 11, size=(total_hours * 60) // period_minutes).tolist()
            
            # default_need = np.random.randint(0, 11, size=(total_hours * 60) // period_minutes).tolist()
            need_input = st.text_area(f"Slot Demand",', '.join(map(str, default_need)), height=210)
            
            #Exibir a quantidade de elementos em default_need
            need = [need.strip() for need in need_input.split(',')] 
            st.write(f"Total Demand {len(need)}")
            
            # Limpar e converter a lista de 'need'
            needNew = [float(x) for x in need if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]

            max_demands_per_driver = st.number_input("Max Demands per Driver", min_value=1)
            limit_workers = st.number_input("Drivers (0, no limit)", min_value=1,  key="limit_workers") #int(math.ceil(sum(needNew) / int(max_demands_per_driver))),  key="limit_workers")
   
    
    # Seleção das restrições
    st.subheader("Restrictions")
    with st.expander("Global", expanded=True):
        
        selected_restrictions = {}
        col1, col2, col3 = st.columns([2, 4, 5])

        # Radiobuttons na primeira coluna
        with col1:
            
            st.write("Objective Function")
            radio_selection_object = st.radio(
                "Select the objective", 
                options=["Maximize Demand Response", "Minimize Total Number of Drivers"],
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
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"], value=default_checked, disabled=True)
                elif restriction["Key"] == "limite_diario":
                    default_checked = restriction["Key"] == "limite_diario"
                    selected_restrictions[restriction["Key"]] = st.checkbox(checkbox_label, key=restriction["Key"], value=default_checked)
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

    st.subheader("Modo de Otimização")

    optimization_mode = st.selectbox(
        "Selecione o modo desejado:",
        ["Exact", "Heuristic", "LNS"],
        index=0
    )

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

    initial_constraints_coefficients = np.zeros((num_periods, num_periods), dtype=int)
    initial_constraints_coefficients = preencher_restricoes(initial_constraints_coefficients, restrictions, selected_restrictions, num_periods, need)

    # Se não houver dados no arquivo, inicializa com uma matriz padrão
    constraints_coefficients = load_data('constraints_coefficients.json') 
    if constraints_coefficients is None or constraints_coefficients.size == 0:
        constraints_coefficients = initial_constraints_coefficients
        save_data(constraints_coefficients, 'constraints_coefficients.json')  # Salva a matriz padrão no arquivo

    st.subheader("Cache")
    with st.expander("Parameters", expanded=True):
        atualizarFicheiro = st.checkbox("Update File", value=False, help="If enabled, the options below will be disabled.")
        if atualizarFicheiro:
            constraints_coefficients = initial_constraints_coefficients
            save_data(initial_constraints_coefficients, 'initial_constraints_coefficients.json')
            save_data(constraints_coefficients, 'constraints_coefficients.json')
        # else:
            # constraints_coefficients = initial_constraints_coefficients

        if num_periods > 0:
            initial_density_matrix = calculate_density(initial_constraints_coefficients)
            initial_density_matrix = initial_density_matrix 
            st.write(f"Initial Density {initial_density_matrix:.2f}%")

    st.subheader("Elementary Operations")
    
    with st.expander("Parameters", expanded=True):
        # Loop para gerar as checkboxes com base nas operações
        selected_operations = {}
        for elementalOperation in elementalOperations:
            checkbox_label = f"{elementalOperation['Description']} | {elementalOperation['Formula']}"
            # checkbox_label = html.escape(f"{elementalOperation['Description']} | {elementalOperation['Formula']}")
            selected_operations[elementalOperation["Key"]] = st.checkbox(
                checkbox_label, 
                key=elementalOperation["Key"], 
                value=False, 
                disabled=atualizarFicheiro,
                help="This option will only be enabled if 'Update File' is disabled."
            )

    # Inicializar variáveis para as operações
    swap_rows = None
    multiply_row = None
    add_multiple_rows = None
    msg = []
    iterations_data_result = []
    try:
        # Sempre cria as colunas ANTES de validar need
        col_resultsItIOpI, col_resultsItIOpII, col_resultsItIOpIII = st.columns(3)
        col_resultsI_col = st.columns(1)[0]
        col_resultsItI_col = st.columns(1)[0]

        need = list(map(int, need_input.split(',')))

        # Exibir a matriz de restrições antes da otimização
        if len(need) != num_periods:
            st.error(f"The input must have exactly {num_periods} values ​​(1 for each period of {period_minutes} minutes).")
        else:
            with col_resultsI_col:
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
                            
                            (solver, 
                            status, 
                            total_workers, 
                            workers_schedule, 
                            constraints_coefficients, 
                            initial_density, 
                            final_density, 
                            statistics_result, 
                            msg, 
                            iterations_data_result, 
                            matrix_allocation,
                            solver_logs
                            ) = run_solver_with_mode(
                                optimization_mode,
                                needNew,
                                variable_type,
                                constraints_coefficients,
                                selected_restrictions,
                                solver_param_type,
                                acceptable_percentage,                            
                                limit_workers,
                                limit_iteration,
                                limit_level_relaxation,
                                max_demands_per_driver,
                                tolerance_demands,
                                penalty,
                                swap_rows=None, 
                                multiply_row=None, 
                                add_multiple_rows=None,
                                radio_selection_object=radio_selection_object                                
                            )                               

                    else:
                        if swap_rows_c:
                            with col_resultsItIOpI:
                                constraints_coefficients = load_data('constraints_coefficients.json') 
                                row1 = st.number_input("Choose line 1 to exchange", min_value=0, max_value=len(constraints_coefficients)-1)
                                row2 = st.number_input("Choose line 2 to exchange:", min_value=1, max_value=len(constraints_coefficients)-1)
                                swap_rows = row1, row2
                                constraints_coefficients = load_data('constraints_coefficients.json')

                                (solver, 
                                status, 
                                total_workers, 
                                workers_schedule, 
                                constraints_coefficients, 
                                initial_density, 
                                final_density, 
                                statistics_result, 
                                msg, 
                                iterations_data_result, 
                                matrix_allocation,
                                solver_logs
                                ) = run_solver_with_mode(
                                    optimization_mode,
                                    needNew,
                                    variable_type,
                                    constraints_coefficients,
                                    selected_restrictions,
                                    solver_param_type,
                                    acceptable_percentage,                            
                                    limit_workers,
                                    limit_iteration,
                                    limit_level_relaxation,
                                    max_demands_per_driver,
                                    tolerance_demands,
                                    penalty,
                                    swap_rows=swap_rows, 
                                    multiply_row=None, 
                                    add_multiple_rows=None, 
                                    radio_selection_object=radio_selection_object                                                               
                                )                            
                        
                        if multiply_row_c:
                            with col_resultsItIOpII:
                                constraints_coefficients = load_data('constraints_coefficients.json') 
                                row = st.number_input("Choose the line to multiply", min_value=0, max_value=len(constraints_coefficients)-1)
                                constant = st.number_input("Choose the constant to multiply", value=1)
                                if constant != 0:
                                    multiply_row = (row, constant)
                                    constraints_coefficients = load_data('constraints_coefficients.json')
                            
                                    (solver, 
                                    status, 
                                    total_workers, 
                                    workers_schedule, 
                                    constraints_coefficients, 
                                    initial_density, 
                                    final_density, 
                                    statistics_result, 
                                    msg, 
                                    iterations_data_result, 
                                    matrix_allocation,
                                    solver_logs
                                    ) = run_solver_with_mode(
                                        optimization_mode,
                                        needNew,
                                        variable_type,
                                        constraints_coefficients,
                                        selected_restrictions,
                                        solver_param_type,
                                        acceptable_percentage,                            
                                        limit_workers,
                                        limit_iteration,
                                        limit_level_relaxation,
                                        max_demands_per_driver,
                                        tolerance_demands,
                                        penalty,
                                        swap_rows=None, 
                                        multiply_row=multiply_row, 
                                        add_multiple_rows=None,
                                        radio_selection_object=radio_selection_object                                                                
                                    )                            
                                else:
                                    st.warning("The multiplication constant cannot be zero!")
                        if add_multiple_rows_c:
                            with col_resultsItIOpIII:
                                constraints_coefficients = load_data('constraints_coefficients.json')
                                row1 = st.number_input("Choose the baseline to sum", min_value=0, max_value=len(constraints_coefficients)-1, key="row1_sum")
                                row2 = st.number_input("Choose the line that will receive the multiple", min_value=0, max_value=len(constraints_coefficients)-1, key="row2_sum")
                                multiple = st.number_input("Choose the multiple to add", value=0, key="multiple_sum")
                                add_multiple_rows = row1, row2, multiple

                                (solver, 
                                status, 
                                total_workers, 
                                workers_schedule, 
                                constraints_coefficients, 
                                initial_density, 
                                final_density, 
                                statistics_result, 
                                msg, 
                                iterations_data_result, 
                                matrix_allocation,
                                solver_logs
                                ) = run_solver_with_mode(
                                    optimization_mode,
                                    needNew,
                                    variable_type,
                                    constraints_coefficients,
                                    selected_restrictions,
                                    solver_param_type,
                                    acceptable_percentage,                            
                                    limit_workers,
                                    limit_iteration,
                                    limit_level_relaxation,
                                    max_demands_per_driver,
                                    tolerance_demands,
                                    penalty,
                                    swap_rows=None, 
                                    multiply_row=None, 
                                    add_multiple_rows=add_multiple_rows,
                                    radio_selection_object=radio_selection_object                                                                
                                )                            

                        if add_multiple_rows_c_auto:
                            
                            constraints_coefficientsNew = load_data('constraints_coefficients.json')
                            constraints_coefficients = load_data('constraints_coefficients.json')
                            #Percorrer a matriz com o multiplicador -1, todas as linhas considerando o row2 sempre um a menor que o row1 e submetendo ao modelo.
                            for idx, row in enumerate(constraints_coefficients):
                                row1 = idx+1
                                row2 = idx
                                multiple = -1
                                add_multiple_rows = row1, row2, multiple
                                
                                # solver, status, total_workers, workers_schedule, constraints_coefficients, initialDensity, finalDensity, statisticsResult, msgResult, iterations_dataResult, matrixAllocation = solve_shift_schedule(
                                #     solverParamType, 
                                #     need, 
                                #     variable_type, 
                                #     constraints_coefficients, 
                                #     selected_restrictions, 
                                #     swap_rows=None, 
                                #     multiply_row=None, 
                                #     add_multiple_rows=add_multiple_rows,
                                #     densidadeAceitavel=acceptable_percentage,
                                #     limit_Workers=limit_Workers,
                                #     limit_Iteration=limit_Iteration,
                                #     limit_Level_Relaxation=limit_Level_Relaxation, 
                                #     max_demands_per_driver=max_demands_per_driver,
                                #     tolerance_Demands=tolerance_Demands, 
                                #     penalty=penalty
                                # )
                                # Depois de montar needNew:
                                # need_numeric = needNew  # só para clareza                                
                                # solver, status, total_workers, workers_schedule, constraints_coefficients, initial_density, final_density, statistics_result, msg_result, iterations_data_result, matrix_allocation = run_solver_with_mode(
                                #         optimization_mode,
                                #         needNew,
                                #         variable_type,
                                #         constraints_coefficients,
                                #         selected_restrictions,
                                #         solver_param_type,
                                #         acceptable_percentage,                            
                                #         limit_workers,
                                #         limit_iteration,
                                #         limit_level_relaxation,
                                #         max_demands_per_driver,
                                #         tolerance_demands,
                                #         penalty,
                                #         swap_rows=None, 
                                #         multiply_row=None, 
                                #         add_multiple_rows=add_multiple_rows,
                                #     )                            

                                (solver, 
                                status, 
                                total_workers, 
                                workers_schedule, 
                                constraints_coefficients, 
                                initial_density, 
                                final_density, 
                                statistics_result, 
                                msg, 
                                iterations_data_result, 
                                matrix_allocation,
                                solver_logs
                                ) = run_solver_with_mode(
                                    optimization_mode,
                                    needNew,
                                    variable_type,
                                    constraints_coefficients,
                                    selected_restrictions,
                                    solver_param_type,
                                    acceptable_percentage,                            
                                    limit_workers,
                                    limit_iteration,
                                    limit_level_relaxation,
                                    max_demands_per_driver,
                                    tolerance_demands,
                                    penalty,
                                    swap_rows=None, 
                                    multiply_row=None, 
                                    add_multiple_rows=add_multiple_rows,
                                    radio_selection_object=radio_selection_object                                                                
                                )                            

                                # Dentro da sua função solve_shift_schedule
                                final_density = calculate_density(constraints_coefficients)
                                
                                # st.write(f"Final density {finalDensity} has reached acceptable limit ({acceptable_percentage}). Exiting loop.")
                                # st.write(f"Test Initial density {calculate_density(initial_constraints_coefficients)}")
                                # st.write(f"Test constraints_coefficients density {calculate_density(constraints_coefficients)}")
                                # st.write(f"Test constraints_coefficientsNew density {calculate_density(constraints_coefficientsNew)}")
                                
                                if final_density <= acceptable_percentage:
                                    st.write(f"Final density {final_density} has reached acceptable limit ({acceptable_percentage}). Exiting loop.")
                                    break
                                # save_data(constraints_coefficients, 'constraints_coefficients.json')

                    # 🔧 Normalização para modos Heuristic/LNS
                    if "workers_schedule" in locals() and "matrix_allocation" in locals():
                        if workers_schedule is None and matrix_allocation is not None:
                            try:
                                workers_schedule = list(np.sum(matrix_allocation, axis=1))
                            except Exception:
                                workers_schedule = []
                                
                             
                                
                # Exibir resultados na primeira coluna
                with col_resultsItI_col:
                        col_resultdetailI,col_resultdetailII,col_resultdetailIII = st.columns(3)
                        if msg is not None:
                            if total_workers is not None:
                                with col_resultdetailI:
                                    # st.subheader("Results")
                                    with st.expander("Results", expanded=True):
                                        
                                        # Processar statisticsResult para separar descrições e valores
                                        results = {
                                            "Description": [],
                                            "Value": []
                                        }
                                        
                                        # Preencher o dicionário com os resultados
                                        if statistics_result is None:
                                            statistics_result = []
                                        for stat in statistics_result:
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
                                with col_resultdetailII:
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
                                with col_resultdetailIII:
                                    
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
                                
                                col_resultsybdetail_I,col_resultsybdetail_II,col_resultsybdetail_III,col_resultsybdetail_IV, col_resultsybdetail_V = st.columns(5)
                                # Gerar DataFrame para os dados
                                slots = list(range(1, len(demanda) + 1))  # Slots de 1 a 96
                                df_comparacao = pd.DataFrame({
                                    "Slot": slots,
                                    "Demanda": demanda,
                                    "Motoristas": workers_schedule
                                })

                                with col_resultsybdetail_I:

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

                                        with st.expander("Análise", expanded=True):
                                            # Exibir as métricas
                                            st.write(f"**Standard Deviation Coverage Rate:** {desvio_padrao:.2f}")
                                        
                                            # Determinar a cor com base no desvio padrão
                                            if desvio_padrao < 0.2:
                                                desvio_cor = "green"
                                                status = "Good"
                                            elif desvio_padrao < 0.5: 
                                                desvio_cor = "orange" 
                                                status = "Plausive"
                                            else:
                                                desvio_cor = "red" 
                                                status = "Poor"

                                            # Exibir o desvio padrão da taxa de cobertura com cor e formatação em negrito
                                            st.markdown(f"<h3 style='color:{desvio_cor}; font-weight: bold; font-size: 16px;'>Standard Deviation Coverage Rate: {desvio_padrao:.2f} ({status})</h3>", unsafe_allow_html=True)

                                        # Calcular e exibir os totais para cada indicador
                                        total_demanda = df_comparacao['Demanda'].sum()
                                        total_motoristas = df_comparacao['Motoristas'].sum()
                                        total_taxa_cobertura = df_comparacao['Taxa de Cobertura'].mean()
                                        total_sobrecarga = df_comparacao['Índice de Sobrecarga'].sum()
                                        total_subutilizacao = df_comparacao['Índice de Subutilização'].sum()
                                    
                                with col_resultsybdetail_II:
                                        with st.expander(f"Demanda vs Motoristas (Demanda Total: {total_demanda}, Motoristas Total: {total_motoristas})", expanded=True):
                                            # Exibir gráfico com barras comparativas
                                            st.bar_chart(df_comparacao.set_index("Slot")[["Demanda", "Motoristas"]])

                                with col_resultsybdetail_III:
                                    with st.expander(f"Taxa Cobertura por Slot (Total: {total_taxa_cobertura:.2f})", expanded=True):
                                        st.bar_chart(df_comparacao.set_index("Slot")['Taxa de Cobertura'])

                                with col_resultsybdetail_IV:
                                    with st.expander(f"Índice Sobrecarga por Slot (Total: {total_sobrecarga:.2f})", expanded=True):
                                        # Gráfico do Índice de Sobrecarga
                                        st.bar_chart(df_comparacao.set_index("Slot")['Índice de Sobrecarga'])

                                with col_resultsybdetail_V:
                                        # Gráfico do Índice de Subutilização
                                    with st.expander(f"Índice Subutilização por Slot (Total: {total_subutilizacao:.2f})", expanded=True):
                                        st.bar_chart(df_comparacao.set_index("Slot")['Índice de Subutilização'])
                                    
            col_resultsIniI, col_resultsIniII = st.columns(2)
            with col_resultsIniI:
                if msg is not None:    
                    if initial_density_matrix is not None:
                        
                        # Exibir a densidade
                        st.write(f"Density: {initial_density_matrix:.4f}")
                        with st.expander("Initial Constraint Matrix", expanded=True):
                            fig, ax = plt.subplots(figsize=(14, 8))
                            sns.heatmap(initial_constraints_coefficients, cmap="Blues", cbar=False, annot=False, fmt="d", annot_kws={"size": 7})
                            plt.title('Constraint Matrix')
                            plt.xlabel('X')
                            plt.ylabel('Period')
                            st.pyplot(fig)
                    
                    # Converter dados para DataFrame
                if iterations_data_result != []:
                    st.subheader("Convergence Progress")
                    df_iterationsResult = pd.DataFrame(iterations_data_result)
                    
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
                        
                        
            # with col_resultsIniII:
            #     if msgResult is None:
            #         #Exibir a densidade
            #         st.write(f"Final Density Matrix Constraints: {final_density:.4f}")
            #         with st.expander("Final Constraints Matrix", expanded=True):
            #             figNew, axNew = plt.subplots(figsize=(14, 8))
            #             constraints_coefficients = load_data('constraints_coefficients.json')
            #             sns.heatmap(constraints_coefficients, cmap="Oranges", cbar=False, annot=False, fmt="d", annot_kws={"size": 6})
            #             plt.title('Constraints Matrix')
            #             plt.xlabel('X')
            #             plt.ylabel('Period')
            #             st.pyplot(figNew)
                        
            with col_resultsIniII:
                if msg is not None:
                    if final_density is not None:
                        st.write(f"Final Density Matrix Constraints: {final_density:.4f}")
                    else:
                        st.write("Final Density Matrix Constraints: not computed for this mode.")
                    with st.expander("Final Constraints Matrix", expanded=True):
                        figNew, axNew = plt.subplots(figsize=(14, 8))
                        constraints_coefficients = load_data('constraints_coefficients.json')
                        sns.heatmap(constraints_coefficients, cmap="Oranges", cbar=False, annot=False, fmt="d", annot_kws={"size": 6})
                        plt.title('Constraints Matrix')
                        plt.xlabel('X')
                        plt.ylabel('Period')
                        st.pyplot(figNew)
                        
                if iterations_data_result != []:
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
            
        
            if iterations_data_result != []:            
                st.table(df_iterationsResult)
            if msg is not None:
                # Criando o DataFrame a partir da lista msgResult
                results_msg_result = {
                    "Description": [],
                    "Value": []
                }
                # Preencher o dicionário com os resultados
                for stat in msg:
                    # Separar a descrição e o valor usando ':'
                    if ':' in stat:
                        descricao, valor = stat.split(':', 1)  # Divide apenas na primeira ocorrência
                        results_msg_result["Description"].append(descricao.strip())  # Adiciona a descrição sem espaços em branco
                        results_msg_result["Value"].append(valor.strip())  # Adiciona o valor sem espaços em branco
                    else:
                        # Caso não haja ':' no stat, adicionar como descrição e valor em branco
                        results_msg_result["Description"].append(stat)
                        results_msg_result["Value"].append("")
                # Criar um DataFrame a partir do dicionário
                dfmsgResult = pd.DataFrame(results_msg_result)
                # Exibir o DataFrame como tabela
                st.table(dfmsgResult)

        with st.expander("Solver Logs", expanded=False):
            st.subheader("Standard Output")

            try:
                stdout_text = solver_logs.get("stdout", "") if isinstance(solver_logs, dict) else str(solver_logs)
                stderr_text = solver_logs.get("stderr", "") if isinstance(solver_logs, dict) else ""
            except Exception:
                stdout_text = str(solver_logs)
                stderr_text = ""

            # Escapar caracteres problemáticos para evitar erros de parser
            stdout_text = stdout_text.replace("<", "&lt;").replace(">", "&gt;")
            stderr_text = stderr_text.replace("<", "&lt;").replace(">", "&gt;")

            st.text_area("stdout", stdout_text, height=250)

            st.subheader("Error Output")
            st.text_area("stderr", stderr_text, height=250)


    except Exception as e:
        # st.error(f"Ocorreu um erro: {e.__class__.__name__}: {e}")
        # st.code(traceback.format_exc(), language="python")
        st.error(f"Ocorreu um erro: {e}")
        st.code(traceback.format_exc())
        


