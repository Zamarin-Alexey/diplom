import math
from random import random

import numpy as np
import pandas as pd

import my_math

from scipy import interpolate
from scipy.optimize import curve_fit

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


class ADFR:
    def __init__(self, sll1_path, sll2_path, col, phi_min, phi_max, approx_mode, poly_degree=10):
        sll1_data = pd.read_csv(sll1_path, sep="\\t", engine="python")
        sll1_data = sll1_data.apply(np.deg2rad)
        x_arr = sll1_data['X'].values

        y1_arr = sll1_data[sll1_data.columns[col]].values

        sll2_data = pd.read_csv(sll2_path, sep="\\t", engine="python")
        sll2_data = sll2_data.apply(np.deg2rad)
        y2_arr = sll2_data[sll2_data.columns[col]].values

        pel_char = [y2_arr[i] - y1_arr[i] for i in range(len(y1_arr))]

        condition = (phi_min < x_arr) & (x_arr < phi_max)
        indices = np.where(condition)
        
        x_arr_trunc = x_arr[indices[0][0]: indices[0][-1] + 1]
        pel_char_trunc = pel_char[indices[0][0]: indices[0][-1] + 1]
        self.G_func = my_math.get_filtered_func(x_arr_trunc, pel_char_trunc, approx_mode, poly_degree)
        
        approxed_pel = [self.G_func(x) for x in x_arr_trunc]
        
        plt.plot(x_arr, y1_arr, label="sll1")
        plt.plot(x_arr, y2_arr, label="sll2")
        plt.plot(x_arr, pel_char, label="pel_char")
        plt.plot(x_arr_trunc, approxed_pel, label="approxed_pel")
        
        plt.legend()
        plt.show()
        
        
    def get_G(self, phi_pel_rad):
        return self.G_func(phi_pel_rad)


class E:
    def __init__(self, U1, U2, phi_pel_deg, K_n, phi_n_deg, noise,
                 omega_c, phi_0, d, lambda_c, ADFR):
        phi_pel_rad = np.deg2rad(phi_pel_deg)
        phi_n_rad = np.deg2rad(phi_n_deg)
        
        G = ADFR.get_G(phi_pel_rad)
        print("G = {}".format(G))
                
        E1, E11, E2, E22 = [], [], [], []
        t_arr = np.arange(0, 10**-9, 10**-12)
        for t in t_arr:
            N1, N2 = noise.get_noise(U1, U2)
            E1.append(U1 * G * math.cos(omega_c * t + phi_0) + N1)
            E11.append(U1 * G * math.sin(omega_c * t + phi_0) + N1)
            E2.append(K_n * U2 * G * math.cos(omega_c * t + phi_0 + phi_n_rad + 2
                                              * pi * d * math.sin(phi_pel_rad) / lambda_c) + N2)
            E22.append(K_n * U2 * G * math.sin(omega_c * t + phi_0 + phi_n_rad + 2
                                               * pi * d * math.sin(phi_pel_rad) / lambda_c) + N2)

        self.E1_func = interpolate.interp1d(t_arr, E1)
        self.E11_func = interpolate.interp1d(t_arr, E11)
        self.E2_func = interpolate.interp1d(t_arr, E2)
        self.E22_func = interpolate.interp1d(t_arr, E22)

        plt.plot(t_arr, self.E1_func(t_arr), label="E1_interp")
        plt.plot(t_arr, self.E11_func(t_arr), label="E11_interp")
        plt.plot(t_arr, self.E2_func(t_arr), label="E2_interp")
        plt.plot(t_arr, self.E22_func(t_arr), label="E22_interp")
        plt.plot(t_arr, np.sqrt(self.E1_func(t_arr)**2+self.E11_func(t_arr)**2), label="E1+E11")
        plt.plot(t_arr, np.sqrt(self.E2_func(t_arr)**2+self.E22_func(t_arr)**2), label="E2+E22")
         
        plt.legend()
        plt.show()
        
    def get_E(self, t):
        return self.E1_func(t), self.E11_func(t), self.E2_func(t), self.E22_func(t)


class DirectionCalculator:
    def __init__(self, q, phi_0_rad, d, K, phi_min_deg, phi_max_deg,
                 lambda_c, sll1_path, sll2_path, approx_mode, freq_num,
                 U1, U2, phi_pel_deg, K_n, phi_n_deg, poly_degree=10):
        self.approx_mode = approx_mode
        self.poly_degree = poly_degree 
        self.phi_min, self.phi_max = (np.deg2rad(phi_min_deg),
                                      np.deg2rad(phi_max_deg))  # диапазон углов
        self.q = q  # сигнал/шум
        self.noise = Noise(q)  # Шум
        self.freq_num = freq_num
        self.ADFR = ADFR(sll1_path, sll2_path, freq_num + 1, self.phi_min, self.phi_max, approx_mode, poly_degree)  # ДНА первой антенны
        self.phi_0 = phi_0_rad  # начальная фаза
        self.d = d  # длина базы
        self.K = K  # пеленгационная чувствительность
        self.lambda_c = lambda_c
        self.omega_c = 2 * pi * C / lambda_c
        self.E = E(U1, U2, phi_pel_deg, K_n, phi_n_deg, self.noise, self.omega_c, phi_0_rad, d,
                   lambda_c, self.ADFR)


    def get_amplitude(self, Ex, Exx):
        return math.sqrt((Ex ** 2) + (Exx ** 2))

    def get_phase(self, Ex, Exx):
        return math.atan2(Exx, Ex)

    def amplitude_method(self):
        t_arr = np.arange(0, 10**-9, 10**-12)
        A1_arr, A2_arr = [], []
        phi_arr = []
        for t in t_arr:
            E1, E11, E2, E22 = self.E.get_E(t)
            A1 = self.get_amplitude(E1, E11)
            A2 = self.get_amplitude(E2, E22)
            A1_arr.append(A1)
            A2_arr.append(A2)
            phi = self.K * ((A1 - A2) / (A1 + A2))
            phi_arr.append(phi)
            
        print(A1_arr[0])
        print(A2_arr[0])
        print(phi_arr[0])
            
        phi_func = my_math.get_filtered_func(t_arr, phi_arr, my_math.FilterMode.POLY, 0)
        phi_approxed = [phi_func(t) for t in t_arr]
        
        plt.plot(t_arr, A1_arr, label="A1")
        plt.plot(t_arr, A2_arr, label="A2")
        plt.plot(t_arr, phi_arr, label="phi")
        plt.plot(t_arr, phi_approxed, label="phi_approxed")
        
        plt.legend()
        plt.show()
        
        return phi_func(t_arr[0])

    def phase_method(self, phase_char_path, approx_mode, poly_degree=10):
        t_arr = np.arange(0, 10**-9, 10**-12)
        phase1_arr, phase2_arr = [], []
        delta_phase_arr = []
        for t in t_arr:
            E1, E11, E2, E22 = self.E.get_E(t)
            phase1 = self.get_phase(E1, E11)
            phase2 = self.get_phase(E2, E22)
            delta_phase = phase2 - phase1
            phase1_arr.append(phase1)
            phase2_arr.append(phase2)
            delta_phase_arr.append(delta_phase)
        delta_phase_normed_arr = [delta_phase / (2 * pi) for delta_phase in delta_phase_arr]
        delta_phase_approxed_func = my_math.get_filtered_func(t_arr, delta_phase_normed_arr, my_math.FilterMode.POLY, 0)
        delta_phase_approxed_arr = [delta_phase_approxed_func(t) for t in t_arr]
        delta_phase = delta_phase_approxed_arr[0]
        plt.plot(t_arr, phase1_arr, label="phase1")
        plt.plot(t_arr, phase2_arr, label="phase2")
        plt.plot(t_arr, delta_phase_arr, label="delta_phase")
        plt.plot(t_arr, delta_phase_normed_arr, label="delta_phase_normed")
        plt.plot(t_arr, delta_phase_approxed_arr, label="delta_phase_approxed")
        plt.legend()
        plt.show()
        
        data = pd.read_csv(phase_char_path, sep='\\t', engine='python')
        data = data.apply(np.deg2rad)
        x_arr = data['X'].values
        faz_arr = data[data.columns[self.freq_num + 1]].values
        shifted_faz = np.unwrap(faz_arr)
        faz_approx_func = my_math.get_filtered_func(x_arr, shifted_faz, approx_mode, poly_degree)
        faz_approxed_arr = [faz_approx_func(x) for x in x_arr]
        faz_approxed_normed_arr = [(faz - pi) % (-2 * pi) + pi for faz in faz_arr]
        plt.plot(x_arr, faz_arr, label="faz")
        plt.plot(x_arr, faz_approxed_arr, label="faz_approxed")
        plt.plot(x_arr, faz_approxed_normed_arr, label="faz_approxed_normed")
        plt.plot(x_arr, [delta_phase for _ in x_arr], label="delta_phase")
        plt.legend()
        plt.show()
        
        print(delta_phase)
        inters = self.find_phase_intersections(delta_phase, faz_approx_func, self.phi_min, self.phi_max)
        i = 0
        for inter in inters:
            plt.plot(x_arr, [inter for _ in x_arr], label="phi"+str(i))
            i += 1
        plt.legend()
        plt.show()
                        
        return inters

    def find_phase_intersections(self, delta_phi, faz_approx_func, phi_min, phi_max):
        eps = 0.0001
        inters = []
        for phi in np.linspace(phi_min, phi_max, 100000):
            if abs((faz_approx_func(phi) - pi) % (-2 * pi) + pi - delta_phi) < eps:
                inters.append(phi)
         
        print(inters)
        for i in range(len(inters) - 1):
            if len(inters) > 1 and abs(inters[i + 1] - inters[i]) < eps:
                inters.pop(i + 1)
        
        return inters

    def choose_right_phi(self, phi_amp: float, phi_phase: set[float]) -> float:
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
