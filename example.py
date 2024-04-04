from direction_calculator import *
import matplotlib.pyplot as plt
import numpy as np

pi = math.pi
q = 14
width = 60
phi_0 = pi / 30
d = 0.15
K = 95
phi_min = -90
phi_max = 90
lambda_c = 9.7 * (10 ** -2)

calc = DirectionCalculator(q, width, phi_0, d, K, phi_min, phi_max, lambda_c)

U1, U2 = 1, 1

phi_pel = 10

K_n = 0.9
phi_n = 0.05

E1_arr, E11_arr, E2_arr, E22_arr = [], [], [], []

t_arr = [i for i in np.arange(0, math.pi, 0.01)]
for t in t_arr:
    E1, E11, E2, E22 = calc.get_E(U1, U2, phi_pel, t, K_n, phi_n)
    E1_arr.append(E1)
    E11_arr.append(E11)
    E2_arr.append(E2)
    E22_arr.append(E22)

plt.plot(t_arr, E1_arr, label=r'E1')
plt.plot(t_arr, E11_arr, label=r'E11')
plt.plot(t_arr, E2_arr, label=r'E2')
plt.plot(t_arr, E22_arr, label=r'E22')
plt.xlabel('t')
plt.ylabel('E')
plt.title('E1, E11, E2, E22')
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
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

plt.plot(t_arr, A1_arr, label=r'A1')
plt.plot(t_arr, A2_arr, label=r'A2')

plt.xlabel('t')
plt.ylabel('A')
plt.title('A')
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.show()

plt.plot(t_arr, phi1_arr, label=r'phi1')
plt.plot(t_arr, phi2_arr, label=r'phi2')
plt.plot(t_arr, delta_phi_arr, label=r'delta_phi')

plt.xlabel('t')
plt.ylabel('phi')
plt.legend(fontsize=16)
plt.title('phi')
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.show()

delta_phi_calc = sum([(i + math.pi) % (2 * math.pi) - math.pi for i in delta_phi_arr]) / len(delta_phi_arr)

angle_arr = [i for i in np.arange(-math.pi / 3, math.pi / 3, 0.01)]
delta_phi_theoretic_arr = []
delta_phi_calc_arr = []
for angle in angle_arr:
    delta_phi_theoretic_arr.append(2 * math.pi * ((d * math.sin(angle) / lambda_c) % 1) - math.pi)
    delta_phi_calc_arr.append(delta_phi_calc)

plt.plot(angle_arr, delta_phi_theoretic_arr, label=r'delta_phi_theoretic')
plt.plot(angle_arr, delta_phi_calc_arr, label=r'delta_phi_calc')
plt.xlabel('phi_pel')
plt.ylabel('delta_phi')
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.show()

probe = calc.phase_method(phi1_arr[0], phi2_arr[0])
phase_angle_arr = [[] for _ in range(len(probe))]
for i in range(len(t_arr)):
    for idx, angle in enumerate(list(calc.phase_method(phi1_arr[i], phi2_arr[i]))):
        phase_angle_arr[idx].append(angle)

amplitude_angle_arr = [calc.amplitude_method(sum(A1_arr) / len(A1_arr), sum(A2_arr) / len(A2_arr)) for _ in t_arr]

for idx, phase_angle in enumerate(phase_angle_arr):
    plt.plot(t_arr, phase_angle, label=f'angle_phase_{idx}')

plt.plot(t_arr, amplitude_angle_arr, label=r'angle_amp')
plt.xlabel('t')
plt.ylabel('angle')
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.show()

print(rad_to_deg(calc.choose_right_phi(calc.amplitude_method(A1_arr[0], A2_arr[0]), calc.phase_method(phi1_arr[0], phi2_arr[0]))))
