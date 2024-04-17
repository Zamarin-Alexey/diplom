import math
from random import random

import numpy as np
import pandas as pd

import my_math

import matplotlib.pyplot as plt

C = 299792458
pi = math.pi


class Noise:
    def __init__(self, q):
        self.q = q

    def get_noise(self, U1, U2):
        A1 = U1 / my_math.db_to_times(self.q)
        A2 = U2 / my_math.db_to_times(self.q)
        N1 = 0 * random() * pi / 2
        N2 = 0 * random() * pi / 2
        return N1, N2


class SLL:
    def __init__(self, path_to_sll, approx_mode, col, poly_degree=10):
        data = pd.read_csv(path_to_sll, sep='\\t', engine='python')
        data = data.apply(np.deg2rad)
        x_arr = data['X'].values
        y_arr = data[data.columns[col]].values
        self.sll_func = my_math.get_filtered_func(x_arr, y_arr, approx_mode, poly_degree)

    def get_sll(self, phi_pel_rad):
        return self.sll_func(phi_pel_rad)


class E:
    def __init__(self, U1, U2, phi_pel_deg, K_n, phi_n_deg, sll1, sll2, noise,
                 omega_c, phi_0, d, lambda_c):
        phi_pel_rad = np.deg2rad(phi_pel_deg)
        G1 = sll1.get_sll(phi_pel_rad)
        G2 = sll2.get_sll(phi_pel_rad)
        phi_n_rad = np.deg2rad(phi_n_deg)
        G = G1 - G2
        E1, E11, E2, E22 = [], [], [], []
        t_arr = np.linspace(0, pi / 6, 100)
        for t in t_arr:
            N1, N2 = noise.get_noise(U1, U2)
            E1.append(U1 * G1 * math.cos(omega_c * t + phi_0) + N1)
            E11.append(U1 * G1 * math.sin(omega_c * t + phi_0) + N1)
            E2.append(K_n * U2 * G2 * math.cos(omega_c * t + phi_0 + phi_n_rad + 2
                                              * pi * d * math.sin(phi_pel_rad) / lambda_c) + N2)
            E22.append(K_n * U2 * G2 * math.sin(omega_c * t + phi_0 + phi_n_rad + 2
                                               * pi * d * math.sin(phi_pel_rad) / lambda_c) + N2)
        mode = my_math.FilterMode.BUTTER
        self.E1_func = my_math.get_filtered_func(t_arr, E1, mode)
        self.E11_func = my_math.get_filtered_func(t_arr, E11, mode)
        self.E2_func = my_math.get_filtered_func(t_arr, E2, mode)
        self.E22_func = my_math.get_filtered_func(t_arr, E22, mode)

    def get_E(self, t):
        return self.E1_func(t), self.E11_func(t), self.E2_func(t), self.E22_func(t)


class DirectionCalculator:
    def __init__(self, q, phi_0_rad, d, K, phi_min_deg, phi_max_deg,
                 lambda_c, sll1_path, sll2_path, approx_mode, freq_num,
                 U1, U2, phi_pel_deg, K_n, phi_n_deg, poly_degree=10):
        self.approx_mode = approx_mode
        self.poly_degree = poly_degree
        self.q = q  # сигнал/шум
        self.noise = Noise(q)  # Шум
        self.freq_num = freq_num
        self.sll1 = SLL(sll1_path, approx_mode, freq_num + 1, poly_degree)  # ДНА первой антенны
        self.sll2 = SLL(sll2_path, approx_mode, freq_num + 1, poly_degree)  # ДНА второй антенны
        self.phi_0 = phi_0_rad  # начальная фаза
        self.d = d  # длина базы
        self.K = K  # пеленгационная чувствительность
        self.phi_min, self.phi_max = (np.deg2rad(phi_min_deg),
                                      np.deg2rad(phi_max_deg))  # диапазон углов
        self.lambda_c = lambda_c
        self.omega_c = 2 * pi * C / lambda_c
        self.E = E(U1, U2, phi_pel_deg, K_n, phi_n_deg, self.sll1, self.sll2, self.noise, self.omega_c, phi_0_rad, d,
                   lambda_c)

    @staticmethod
    def get_amplitude_and_phase(Ex, Exx):
        A = math.sqrt((Ex ** 2) + (Exx ** 2))
        phi = math.atan2(Exx, Ex)
        return A, phi

    def amplitude_method(self, A1, A2):
        A = self.K * ((A1 - A2) / (A1 + A2))
        return A

    def phase_method(self, delta_phi, phase_char_path):
        data = pd.read_csv(phase_char_path, sep='\\t', engine='python')
        data = data.apply(np.deg2rad)
        x_arr = data['X'].values
        y_arr = data[data.columns[self.freq_num + 1]].values
        k, b = my_math.find_phase_line_coeffs(x_arr, y_arr)
        return my_math.find_phase_intersections(k, b, delta_phi, self.phi_min, self.phi_max)

    @staticmethod
    def choose_right_phi(phi_amp: float, phi_phase: set[float]) -> float:
        print(phi_amp)
        best = None
        best_diff = None
        for phi in phi_phase:
            if best_diff is None or abs(phi - phi_amp) < best_diff:
                best = phi
                best_diff = abs(phi - phi_amp)
                continue
            break
        return best

    def calculate_accuracy(self, phi):
        return ((self.q ** 0.5) * (2 * pi * self.d / self.lambda_c) * math.cos(phi)) ** -1
