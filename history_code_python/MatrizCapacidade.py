n = 96  # Número de períodos

# Abrir um arquivo para salvar a matriz
with open("matriz_capacidades.txt", "w") as file:
    for i in range(1, n+1):
        for j in range(1, n+1):
            file.write(f"{i} {j} 10,\n")

print("Matriz de capacidades gerada e salva em 'matriz_capacidades.txt'.")
