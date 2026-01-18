import streamlit as st
from models import create_solver, define_model
from utils import validate_input, plot_allocation, generate_pdf_report

def main():
    st.title("Otimização de Alocação de Motoristas")
    
    # Entrada do usuário
    st.sidebar.title("Configuração do Modelo")
    num_slots = st.sidebar.slider("Número de Slots", 24, 360, step=24)
    demand = st.sidebar.text_area("Demanda por Slot (separada por vírgula)", "1,2,3,4")
    demand = [int(d) for d in demand.split(",")]

    # Validação dos dados
    if validate_input(num_slots, demand):
        st.success("Dados válidos!")
        max_slots_per_driver = 36
        solver = create_solver()
        
        # Definição do modelo
        solver, x = define_model(solver, demand, num_slots, max_slots_per_driver)
        status = solver.Solve()

        if status == solver.OPTIMAL:
            st.success("Solução ótima encontrada!")
            total_drivers = int(solver.Objective().Value())
            st.write(f"Total de motoristas necessários: {total_drivers}")

            # Preparar resultados
            solution = [int(x[t].solution_value()) for t in range(num_slots)]
            
            # Visualizar alocação
            plot_allocation(solution, demand, num_slots)
            
            # Gerar relatório PDF
            if st.button("Gerar Relatório em PDF"):
                generate_pdf_report(solution, demand, total_drivers)
                st.success("Relatório gerado: 'report.pdf'")
        else:
            st.error("Solução não encontrada! Verifique os dados.")
    else:
        st.error("Dados inválidos! Verifique as configurações e tente novamente.")

if __name__ == "__main__":
    main()
