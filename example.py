from direction_calculator import *
import matplotlib.pyplot as plt
import numpy as np

q = 100
width_deg = 60
phi_0 = 6
d = 0.12
K = 95
phi_min = - math.pi / 3
phi_max = math.pi / 3

calc = DirectionCalculator(q, width_deg, phi_0, d, K, phi_min, phi_max)

U1, U2 = 5, 5

phi_pel = 12.5

K_n = 0.9
phi_n = 0.05

E1_arr, E11_arr, E2_arr, E22_arr = [], [], [], []

t_arr = [i for i in np.arange(0, math.pi, 0.01)]
for t in t_arr:
    E1, E11,E2, E22 = calc.get_E(U1, U2, phi_pel, t, K_n, phi_n)
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
for E1, E11, E2, E22 in zip(E1_arr, E11_arr, E2_arr, E22_arr):
    A1, phi1 = calc.get_amplitude_and_phase(E1, E11)
    A1_arr.append(A1)
    phi1_arr.append(phi1)
    A2, phi2 = calc.get_amplitude_and_phase(E2, E22)
    A2_arr.append(A2)
    phi2_arr.append(phi2)

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


amplitude_angle = calc.amplitude_method(sum(A1_arr)/len(A1_arr), sum(A2_arr)/len(A2_arr))
phase_angle = calc.phase_method(sum(phi1_arr)/len(phi1_arr), sum(phi2_arr)/len(phi2_arr))

angle_arr = [i for i in np.arange(-math.pi/3, math.pi/3, 0.01)]
delta_phi_theoretic_arr = []
delpta_phi_calc_arr = []
for angle in angle_arr:
    delta_phi_theoretic_arr.append(2*math.pi(d*sin(pi/180*angle)/lambda_c))
    delpta_phi_calc_arr.append(phase_angle)
    
plt.plot(angle_arr, delta_phi_theoretic_arr, label=r'delta_phi_theoretic')
plt.plot(angle_arr, delpta_phi_calc_arr, label=r'delta_phi_calc')
plt.xlabel('phi_pel')
plt.ylabel('delta_phi')
plt.legend(fontsize=16)
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.show()

