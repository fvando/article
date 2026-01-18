import numpy as np

# Sistema de equações representado pela matriz de coeficientes
# Exemplo: 2x + 3y = 5  -> [2, 3, 5]
#          4x + y = 11   -> [4, 1, 11]
constraint_matrix = np.array([[2, 3, 5],
                               [4, 1, 11]])

print("Matriz Inicial")
print(constraint_matrix)

# 1. Trocar as linhas
constraint_matrix[[0, 1]] = constraint_matrix[[1, 0]]
print("Após trocar as equações:")
print(constraint_matrix)

# 2. Multiplicar a primeira equação por 2
constraint_matrix[0] *= 2
print("Após multiplicar a primeira equação por 2:")
print(constraint_matrix)

# 3. Somar 3 vezes a primeira equação à segunda
constraint_matrix[1] += 3 * constraint_matrix[0]
print("Após somar 3 vezes a primeira equação à segunda:")
print(constraint_matrix)
