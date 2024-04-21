import numpy as np

# Ваш массив
array = np.array([1, 2, 3, 4, 5, 6])

# Условие
condition = array > 3

# Получение индексов, которые удовлетворяют условию
indices = np.where(condition)

print(indices)
