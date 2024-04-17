import my_math
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

pi = math.pi

data = pd.read_csv("pel_data/faz.txt", sep="\\t", engine="python")
data = data.apply(np.deg2rad)
x_arr = data["X"].values
y_arr = data[data.columns[3]].values

k, b = my_math.find_phase_line_coeffs(x_arr, y_arr)
b = b % pi
y = pi
x_vals = my_math.find_phase_intersections(k, b, y, -pi / 4, pi / 4)
print(x_vals)
new_arr = [((k * x + b) - pi) % (-2 * pi) + pi for x in x_arr]
y_val = [y for _ in x_arr]

plt.plot(x_arr, new_arr, linestyle="--")
plt.plot(x_arr, y_arr)
plt.plot(x_arr, y_val)
plt.show()
