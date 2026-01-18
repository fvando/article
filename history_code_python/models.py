from ortools.linear_solver import pywraplp

def create_solver():
    """Inicializa o solver do OR-Tools."""
    return pywraplp.Solver.CreateSolver('SCIP')

def define_model(solver, needs, time_slots, max_slots_per_driver):
    """
    Define o modelo matemático para o problema de alocação de motoristas.
    - needs: lista com necessidades por slot.
    - time_slots: número total de slots.
    - max_slots_per_driver: limite de slots que um motorista pode atender.
    """
    # Variáveis de decisão
    x = {}
    for t in range(time_slots):
        x[t] = solver.BoolVar(f'x_{t}')
    
    # Restrições de cobertura de demanda
    for t in range(time_slots):
        solver.Add(solver.Sum([x[t]]) >= needs[t])

    # Restrições de limite diário
    solver.Add(solver.Sum([x[t] for t in range(time_slots)]) <= max_slots_per_driver)
    
    # Função objetivo: Minimizar motoristas
    solver.Minimize(solver.Sum([x[t] for t in range(time_slots)]))
    return solver, x
