import math
from direction_calculator import *  

q = 14.0
phi_0 = 6.0
phi_min = -35.0
phi_max = 35.0
lambda_c = 0.097
sll1_path = "pel_data/12.txt"
sll2_path = "pel_data/13.txt"
faz_path = "pel_data/faz.txt"
approx_mode = my_math.ApproxMode.POLY
freq_num = 3
poly_degree = 10
phi_pel = 0.0
K_n = 0.9
phi_n = 0.05
U1, U2 = 5.0, 5.0
noise_enable = False

calc = DirectionCalculator(
    q,
    phi_0,
    phi_min,
    phi_max,
    lambda_c,
    sll1_path,
    sll2_path,
    faz_path,
    approx_mode,
    freq_num,
    U1,
    U2,
    phi_pel,
    K_n,
    phi_n,
    noise_enable,
    poly_degree,
    None,
    None,
    None,
    None,
    None,
)

calc.amplitude_method()
print(calc.phase_method())
