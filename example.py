import my_math
from direction_calculator import *
import matplotlib.pyplot as plt
import numpy as np

pi = math.pi
q = 14
width = 60
phi_0 = pi / 30
d = 0.15
K = 95
phi_min = -45
phi_max = 45
lambda_c = 9.7 * (10**-2)
sll1_path = "pel_data/12.txt"
sll2_path = "pel_data/13.txt"
approx_mode = my_math.FilterMode.POLY
freq_num = 1
poly_degree = 20
phi_pel = 35
K_n = 0.9
phi_n = 0.05
U1, U2 = 1, 1

calc = DirectionCalculator(
    q,
    phi_0,
    d,
    K,
    phi_min,
    phi_max,
    lambda_c,
    sll1_path,
    sll2_path,
    approx_mode,
    freq_num,
    U1,
    U2,
    phi_pel,
    K_n,
    phi_n,
    poly_degree,
)

E1_arr, E11_arr, E2_arr, E22_arr = [], [], [], []

t_arr = np.linspace(0, pi / 6, 1000)
for t in t_arr:
    E1, E11, E2, E22 = calc.E.get_E(t)
    E1_arr.append(E1)
    E11_arr.append(E11)
    E2_arr.append(E2)
    E22_arr.append(E22)

plt.plot(t_arr, E1_arr, label=r"E1")
plt.plot(t_arr, E11_arr, label=r"E11")
plt.plot(t_arr, E2_arr, label=r"E2")
plt.plot(t_arr, E22_arr, label=r"E22")
plt.xlabel("t")
plt.ylabel("E")
plt.title("E1, E11, E2, E22")
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.tight_layout()
plt.show()

A1_arr, phi1_arr = [], []
A2_arr, phi2_arr = [], []
delta_phi_arr = []

for E1, E11, E2, E22 in zip(E1_arr, E11_arr, E2_arr, E22_arr):
    A1, phi1 = calc.get_amplitude_and_phase(E1, E11)
    A1_arr.append(A1)
    phi1_arr.append(phi1)
    A2, phi2 = calc.get_amplitude_and_phase(E2, E22)
    A2_arr.append(A2)
    phi2_arr.append(phi2)
    delta_phi_arr.append(phi2 - phi1)

A1_func = my_math.get_filtered_func(t_arr, A1_arr, my_math.FilterMode.POLY, 0)
A2_func = my_math.get_filtered_func(t_arr, A2_arr, my_math.FilterMode.POLY, 0)
A1 = [A1_func(t) for t in t_arr]
A2 = [A2_func(t) for t in t_arr]

plt.plot(t_arr, A1_arr, label=r"A1")
plt.plot(t_arr, A2_arr, label=r"A2")
plt.plot(t_arr, A1, label=r"approxed_A1")
plt.plot(t_arr, A2, label=r"approxed_A2")

plt.xlabel("t")
plt.ylabel("A")
plt.title("A")
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.tight_layout()
plt.show()

k1, b1 = my_math.find_phase_line_coeffs(t_arr, phi1_arr)
k2, b2 = my_math.find_phase_line_coeffs(t_arr, phi2_arr)
phi1_arr_approxed = [((k1 * t + b1) - pi) % (-2 * pi) + pi for t in t_arr]
phi2_arr_approxed = [((k2 * t + b2) - pi) % (-2 * pi) + pi for t in t_arr]
delta_phi_approxed = [
    phi2_arr_approxed[i] - phi1_arr_approxed[i] for i in range(len(phi1_arr_approxed))
]

plt.plot(t_arr, phi1_arr, label=r"phi1")
plt.plot(t_arr, phi2_arr, label=r"phi2")
plt.plot(t_arr, delta_phi_arr, label=r"delta_phi")
plt.plot(t_arr, phi1_arr_approxed, label=r"phi1_approxed", linestyle="--")
plt.plot(t_arr, phi2_arr_approxed, label=r"phi2_approxed", linestyle="--")
plt.plot(t_arr, delta_phi_approxed, label=r"delta_phi_approxed")


plt.xlabel("t")
plt.ylabel("phi")
plt.legend(fontsize=16)
plt.title("phi")
plt.minorticks_on()
plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.tight_layout()
plt.show()

data = pd.read_csv("pel_data/faz.txt", sep="\\t", engine="python")
data = data.apply(np.deg2rad)
x_arr = data["X"].values
y_arr = data[data.columns[freq_num + 1]].values
k, b = my_math.find_phase_line_coeffs(x_arr, y_arr)
faz_approxed = [((k * x + b) - pi) % (-2 * pi) + pi for x in x_arr]

delta_phi = delta_phi_approxed[0]
delta_phi_norm = delta_phi_approxed[0]
if delta_phi > pi:
    delta_phi_norm = -2 * pi + delta_phi
if delta_phi < -pi:
    delta_phi_norm = 2 * pi - delta_phi
delta_phi_norm_arr = [delta_phi_norm for x in x_arr]

plt.plot(x_arr, y_arr, label=r"faz")
plt.plot(x_arr, faz_approxed, label=r"faz_approxed")
plt.plot(x_arr, delta_phi_norm_arr, label=r"delta_phi")

plt.xlabel("phi_pel")
plt.ylabel("delta_phi")
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.tight_layout()
plt.show()

inters = calc.phase_method(delta_phi, "pel_data/faz.txt")
print(inters)

i = 0
for inter in inters:
    plt.plot(t_arr, [inter for _ in t_arr], label=i)
    i += 1


amplitude_angle_arr = [calc.amplitude_method(A1_arr[i], A2_arr[i]) for i in range(len(t_arr))]
plt.plot(t_arr, amplitude_angle_arr, label=r"angle_amp")
plt.xlabel("t")
plt.ylabel("angle")
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.tight_layout()
plt.show()

print(
    np.rad2deg(
        calc.choose_right_phi(calc.amplitude_method(A1_arr[0], A2_arr[0]), inters)
    )
)
