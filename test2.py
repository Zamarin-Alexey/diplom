import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import random
import math

# Создаем пилообразный сигнал
x = np.linspace(0, 10*np.pi, 500)
y = [math.sin(x*1000)/100 + 0.01 for x in x]

# Добавляем шум

yn = [y + (random.random() - 0.5)/200 for y in y]

# Применяем фильтр Savitzky-Golay
ys = savgol_filter(yn, 50, 10)

# Рисуем графики
plt.plot(x, y, label='Изначальный сигнал')
plt.plot(x, yn, label='Зашумленный сигнал')
plt.plot(x, ys, label='Сглаженный сигнал')
plt.legend()
plt.show()
