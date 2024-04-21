import math
from direction_calculator import *

pi = math.pi
q = 14
width = 60
phi_0 = pi / 30
d = 0.15
K = 0.95
phi_min = -45
phi_max = 45
lambda_c = 9.7 * (10**-2)
sll1_path = "pel_data/12.txt"
sll2_path = "pel_data/13.txt"
approx_mode = my_math.FilterMode.POLY
freq_num = 6
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

# print(calc.amplitude_method())
print(calc.phase_method("pel_data/faz.txt", approx_mode, poly_degree))
