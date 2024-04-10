import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Load the data from the file
data = pd.read_csv('pel_data/faz.txt', sep='\\t')
data = data.apply(np.deg2rad)
x = data['X'].values
y = [data[column].values for column in data.columns[1:]]

# Define a function to shift the Y values
def shift_y_values(y_values):
    shifted_y_values = []
    for y in y_values:
        shifted_y = np.unwrap(y)
        shifted_y_values.append(shifted_y)
    return shifted_y_values

# Shift the Y values
shifted_y_values = shift_y_values(y_values)

# Define a function to perform linear interpolation
def interpolate_y_values(x, y_values):
    interpolated_y_values = []
    for y in y_values:
        func = np.poly1d(np.polyfit(x, y, 1))
        interpolated_y = np.vectorize(lambda x: func(x))(x)
        interpolated_y_values.append(interpolated_y)
    return interpolated_y_values

# Perform linear interpolation
interpolated_y_values = interpolate_y_values(x, shifted_y_values)

# Shift the interpolated Y values back
shifted_back_y_values = [np.angle(np.exp(1j*y)) for y in interpolated_y_values]

# Convert radians back to degrees
shifted_back_y_values = [np.rad2deg(y) for y in shifted_back_y_values]

# Plot the original and approximated functions
plt.figure(figsize=(10, 5))
for i in range(len(y_values)):
    plt.plot(x, np.rad2deg(y_values[i]), label=f'Original {i+1}')
    plt.plot(x, shifted_back_y_values[i], label=f'Approximated {i+1}')
    plt.legend()
    plt.plot(x, np.rad2deg(interpolated_y_values[i]))
    plt.show()


def func(x, k, b):
    return k * x + b


k = 10
b = 5

x_arr = np.arange(-math.pi / 4, math.pi / 4, 0.001)
y_arr = [(func(x, k, b) + math.pi) % (2 * math.pi) - math.pi for x in x_arr]

delta_phi = math.pi / 6

inters = my_math.find_phase_intersections(k, b, delta_phi, -math.pi / 4, math.pi / 4)
delta_phi_arr = [delta_phi for _ in x_arr]
print(inters)

plt.plot(x_arr, y_arr)
plt.plot(x_arr, delta_phi_arr)
plt.show()


