import my_math
from direction_calculator import *
import matplotlib.pyplot as plt
import numpy as np

sll1_path = "pel_data/12.txt"
sll2_path = "pel_data/13.txt"
col = 5
bounds = pi/8

sll1_data = pd.read_csv(sll1_path, sep="\\t", engine="python")
sll1_data = sll1_data.apply(np.deg2rad)
x_arr = sll1_data['X'].values


y1_arr = sll1_data[sll1_data.columns[col]].values

sll2_data = pd.read_csv(sll2_path, sep="\\t", engine="python")
sll2_data = sll2_data.apply(np.deg2rad)
y2_arr = sll2_data[sll2_data.columns[col]].values

pel_char = [y2_arr[i] - y1_arr[i] for i in range(len(y1_arr))]

condition = (-bounds < x_arr) & (x_arr < bounds)
indices = np.where(condition)
x_arr_trunc = x_arr[indices[0][0]: indices[0][-1] + 1]
pel_char_trunc = pel_char[indices[0][0]: indices[0][-1] + 1]
butter_func = my_math.get_filtered_func(x_arr_trunc, pel_char_trunc, my_math.FilterMode.BUTTER)
savgol_func = my_math.get_filtered_func(x_arr_trunc, pel_char_trunc, my_math.FilterMode.SAVGOL)
poly_func_5 = my_math.get_filtered_func(x_arr_trunc, pel_char_trunc, my_math.FilterMode.POLY, 5)
poly_func_10 = my_math.get_filtered_func(x_arr_trunc, pel_char_trunc, my_math.FilterMode.POLY, 10)
poly_func_20 = my_math.get_filtered_func(x_arr_trunc, pel_char_trunc, my_math.FilterMode.POLY, 20)
line_func = my_math.get_filtered_func(x_arr_trunc, pel_char_trunc, my_math.FilterMode.POLY, 1)

butter = [butter_func(x) for x in x_arr_trunc]
savgol = [savgol_func(x) for x in x_arr_trunc]
poly_5 = [poly_func_5(x) for x in x_arr_trunc]
poly_10 = [poly_func_10(x) for x in x_arr_trunc]
poly_20 = [poly_func_20(x) for x in x_arr_trunc]
line = [line_func(x) for x in x_arr_trunc]

plt.plot(x_arr, y1_arr, label="sll1")
plt.plot(x_arr, y2_arr, label="sll2")
plt.plot(x_arr, pel_char, label="pel_char")
# plt.plot(x_arr_trunc, butter, label="butter")
# plt.plot(x_arr_trunc, savgol, label="savgol")
plt.plot(x_arr_trunc, poly_5, label="poly_5")
plt.plot(x_arr_trunc, poly_10, label="poly_10")
# plt.plot(x_arr_trunc, poly_20, label="poly_20")
plt.plot(x_arr_trunc, line, label="line")

plt.legend()
plt.show()

