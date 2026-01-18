import pulp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Função para aplicar operações elementares
def apply_elementary_operations(constraint_matrix):
    print("\nOperações Elementares nas Restrições")

    swap_rows = input("Troca de Equações? (s/n): ").lower() == 's'
    multiply_row = input("Multiplicação por Constante? (s/n): ").lower() == 's'
    add_multiple_rows = input("Somar Múltiplo de uma Equação a Outra? (s/n): ").lower() == 's'
    
    if swap_rows:
        row1 = int(input("Escolha a linha 1 para trocar: "))
        row2 = int(input("Escolha a linha 2 para trocar: "))
        constraint_matrix[[row1, row2]] = constraint_matrix[[row2, row1]]
        print(f"Linhas {row1} e {row2} trocadas com sucesso!")
    
    if multiply_row:
        row = int(input("Escolha a linha para multiplicar: "))
        constant = float(input("Escolha a constante para multiplicar: "))
        if constant != 0:
            constraint_matrix = constraint_matrix.astype(np.float64)
            constraint_matrix[row] *= constant
            print(f"Linha {row} multiplicada por {constant} com sucesso!")
        else:
            print("A constante de multiplicação não pode ser zero!")

    if add_multiple_rows:
        row1 = int(input("Escolha a linha base para somar: "))
        row2 = int(input("Escolha a linha que vai receber o múltiplo: "))
        multiple = float(input("Escolha o múltiplo para somar: "))
        constraint_matrix = constraint_matrix.astype(np.float64)
        constraint_matrix[row2] += multiple * constraint_matrix[row1]
        print(f"Múltiplo da linha {row1} somado à linha {row2} com sucesso!")

    print("\nMatriz de restrições após as operações:")
    print(constraint_matrix)

    return constraint_matrix

# Função de otimização
def solve_shift_schedule(need, variable_type, solver_choice, constraint_matrix):
    num_periods = len(need)
    prob = pulp.LpProblem("Shift_Scheduling", pulp.LpMinimize)
    
    # Mapeando o tipo de variável escolhido pelo usuário
    cat = pulp.LpContinuous if variable_type == "Contínua" else (pulp.LpBinary if variable_type == "Binária" else pulp.LpInteger)
    
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

    print("\nMatriz de Restrições Inicial:")
    print(constraint_matrix)
    
    # Selecionar o solver com base na escolha do usuário
    solver = None
    if solver_choice == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=True)
    elif solver_choice == "GLPK":
        solver = pulp.GLPK_CMD(msg=True)
    elif solver_choice == "CPLEX":
        solver = pulp.CPLEX_CMD(msg=True)
    elif solver_choice == "Gurobi":
        solver = pulp.GUROBI_CMD(msg=True)
    else:
        print("Solver inválido. Selecione um solver disponível.")
        return None, None, None, None, None, None, None

    # Resolver o problema
    prob.solve(solver)

    # Extrair resultados
    workers_schedule = [int(pulp.value(X[d])) for d in range(num_periods)]
    total_workers = pulp.value(prob.objective)
    
    # Calcular densidade
    density = np.sum(constraint_matrix) / (num_periods ** 2)
    
    # Exibir os resultados
    print("\nResultados")
    print(f"Total de trabalhadores necessários: {total_workers}")

    # Escalonamento dos trabalhadores
    print("Escalonamento dos trabalhadores (1 significa que um trabalhador começa neste período):")
    print(workers_schedule)

    # Visualização dos dados
    plt.bar(range(num_periods), workers_schedule)
    plt.title('Escalonamento dos Trabalhadores')
    plt.xlabel('Período')
    plt.ylabel('Número de Trabalhadores')
    plt.show()

    # Exibir a matriz de restrições
    print("\nMatriz de Restrições:")
    plt.figure(figsize=(10, 6))
    sns.heatmap(constraint_matrix, cmap="Blues", cbar=True)
    plt.title("Matriz de Restrições (1 = Cobertura do Período, 0 = Não Coberto)")
    plt.xlabel("Período")
    plt.ylabel("Trabalhador")
    plt.show()

    print(f"Densidade da matriz de restrições: {density:.4f}")
    print("Densidade é a proporção de restrições ativas em relação ao total possível de restrições.")

    return total_workers, workers_schedule, constraint_matrix, density

# Interface do Console
print("24-Hour Shift Scheduler")
total_hours = int(input("Quantidade total de horas: "))
period_minutes = int(input("Duração de cada período (em minutos): "))

# Calcular o número total de períodos
num_periods = (total_hours * 60) // period_minutes

# Gerar demanda aleatória entre 0 e 10 para cada período
default_need = np.random.randint(0, 11, size=num_periods).tolist()
need_input = input(f"Demanda de trabalhadores (separado por vírgulas para cada período de {period_minutes} minutos): (deixe vazio para usar o padrão: {default_need}) ")
if need_input == "":
    need_input = ', '.join(map(str, default_need))
need = list(map(int, need_input.split(',')))

# Seleção do tipo de variável
variable_type = input("Escolha o tipo de variável (Contínua, Binária, Inteira): ")

# Seleção do solver
solver_choice = input("Escolha o solver (CBC, GLPK, CPLEX, Gurobi): ")

# Matriz de restrições
constraint_matrix = np.zeros((num_periods, num_periods), dtype=int)

# Preencher a matriz de restrições de acordo com os períodos cobertos
for i in range(num_periods):
    for j in range(num_periods):
        if (0 <= (i - j) % num_periods < 18) or (19 <= (i - j) % num_periods < 23) or (25 <= (i - j) % num_periods < 43):
            constraint_matrix[i, j] = 1  # Atualizando a matriz de restrições com os valores de cobertura

# Aplicar operações elementares à matriz de restrições **antes** da execução
constraint_matrix = apply_elementary_operations(constraint_matrix)

# Resolver o problema de otimização
total_workers, workers_schedule, constraint_matrix, density = solve_shift_schedule(need, variable_type, solver_choice, constraint_matrix)
