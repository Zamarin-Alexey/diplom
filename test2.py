from direction_calculator import Noise
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import my_math

data = pd.read_csv("./pel_data/12.txt", sep="\\t", engine="python")
        
x_arr = data['X'].values

y_arr = data[data.columns[4]].values

mode = my_math.ApproxMode()

f = my_math.get_approx_func(x_arr, y_arr, mode, 20)

y1 = [f(x) for x in x_arr]

plt.plot(x_arr, y_arr, label="Чистый массив")
plt.plot(x_arr, y1, label="Аппроксимация")
plt.legend()
plt.show()

