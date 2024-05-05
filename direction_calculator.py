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
    def __init__(self, q, enable):
        self.q = q
        self.enable = enable

    def get_noise(self, U1, U2):
        if self.enable:
            A1 = math.sqrt(my_math.db_to_times(self.q))*U1
            A2 = math.sqrt(my_math.db_to_times(self.q))*U2
        else:
            A1, A2 = 0, 0
        N1 = A1 * random()
        N2 = A2 * random()
        return N1, N2


class ADFR:
    def __init__(self, sll1_path, sll2_path, col, phi_min, phi_max, approx_mode, poly_degree, canvas):
        sll1_data = pd.read_csv(sll1_path, sep="\\t", engine="python")
        
        x_arr = sll1_data['X'].values

        y1_arr = sll1_data[sll1_data.columns[col]].values

        sll2_data = pd.read_csv(sll2_path, sep="\\t", engine="python")
        y2_arr = sll2_data[sll2_data.columns[col]].values
        phi_min_deg = np.rad2deg(phi_min)
        phi_max_deg = np.rad2deg(phi_max)
        pel_char = [y1_arr[i] - y2_arr[i] for i in range(len(y1_arr))]
        condition = (phi_min_deg < x_arr) & (x_arr < phi_max_deg)
        indices = np.where(condition)
        
        x_arr_trunc = x_arr[indices[0][0]: indices[0][-1] + 1]
        pel_char_trunc = pel_char[indices[0][0]: indices[0][-1] + 1]
        self.G_func = my_math.get_approx_func(x_arr_trunc, pel_char_trunc, approx_mode, poly_degree)
        
        self.G1 = my_math.get_approx_func(x_arr_trunc, y1_arr[indices[0][0]: indices[0][-1] + 1], approx_mode, poly_degree)
        self.G2 = my_math.get_approx_func(x_arr_trunc, y2_arr[indices[0][0]: indices[0][-1] + 1], approx_mode, poly_degree)
        
        self.reversed_G_func = interpolate.interp1d(self.G_func(x_arr_trunc), x_arr_trunc)
        
        approxed_pel = [self.G_func(x) for x in x_arr_trunc]
        
        canvas.axs.cla()
                
        canvas.axs.plot(x_arr, y1_arr, label="sll1")
        canvas.axs.plot(x_arr, y2_arr, label="sll2")
        canvas.axs.plot(x_arr, np.array(pel_char), label="pel_char")
        canvas.axs.plot(x_arr_trunc, np.array(approxed_pel), label="approxed_pel")
        canvas.axs.legend()
        canvas.draw()
        
        # plt.plot(x_arr, y1_arr, label="sll1")
        # plt.plot(x_arr, y2_arr, label="sll2")
        # plt.plot(x_arr, pel_char, label="pel_char")
        # plt.plot(x_arr_trunc, approxed_pel, label="approxed_pel")
        # plt.legend()
        # plt.show()
        
    def get_G(self, phi_pel_rad):
        return self.G_func(phi_pel_rad)
    
    def get_G1(self, phi_pel_rad):
        return self.G1(phi_pel_rad)
    
    def get_G2(self, phi_pel_rad):
        return self.G2(phi_pel_rad)
    
    def get_phi(self, G):
        return self.reversed_G_func(G)
    
class PDFR:
    def __init__(self, faz_path, col, approx_mode, poly_degree=10):
        data = pd.read_csv(faz_path, sep='\\t', engine='python')
        data = data.apply(np.deg2rad)
        x_arr = data['X'].values
        faz_arr = data[data.columns[col]].values
        
        zero_idx = 0
        while x_arr[zero_idx] < 0:
            zero_idx += 1
        offset = 0

        for i in range(zero_idx, 0, -1):
            if faz_arr[i+1] >= 0 and faz_arr[i] < 0:
                offset += 1

        shifted_faz = np.unwrap(faz_arr)
        shifted_faz_with_offset = [shifted + offset*2*pi for shifted in shifted_faz]
        self.faz_approx_func = my_math.get_approx_func(x_arr, shifted_faz_with_offset, approx_mode, poly_degree)
    
    def get_faz(self, x):
        return self.faz_approx_func(x)
    
    def get_normed_faz(self, x):
        return (self.faz_approx_func(x) - pi) % (-2 * pi) + pi
    

class E:
    def __init__(self, phi_pel_deg, K_n, phi_n_deg, noise,
                 omega_c, phi_0_deg, ADFR, PDFR, canvas):
        phi_n_rad = np.deg2rad(phi_n_deg)
        phi_0_rad = np.deg2rad(phi_0_deg)
        
        B1 = ADFR.get_G1(phi_pel_deg)
        B2 = ADFR.get_G2(phi_pel_deg)
        U1 = 10 ** (B1/20)
        U2 = 10 ** (B2/20)

        phi_pel_rad = np.deg2rad(phi_pel_deg)
        faz = PDFR.get_normed_faz(phi_pel_rad)
                
        E1, E11, E2, E22 = [], [], [], []
        t_arr = np.arange(0, 10**-9, 10**-12)
        for t in t_arr:
            N1, N2 = noise.get_noise(U1, U2)
            E1.append(U1 * math.cos(omega_c * t + phi_0_rad) + N1)
            E11.append(U1 * math.sin(omega_c * t + phi_0_rad) + N1)
            E2.append(K_n * U2 * math.cos(omega_c * t + phi_0_rad + phi_n_rad + faz) + N2)
            E22.append(K_n * U2 * math.sin(omega_c * t + phi_0_rad + phi_n_rad + faz) + N2)

        self.E1_func = interpolate.interp1d(t_arr, E1)
        self.E11_func = interpolate.interp1d(t_arr, E11)
        self.E2_func = interpolate.interp1d(t_arr, E2)
        self.E22_func = interpolate.interp1d(t_arr, E22)

        canvas.axs.cla()
        
        canvas.axs.plot(t_arr, self.E1_func(t_arr), label="E1_interp")
        canvas.axs.plot(t_arr, self.E11_func(t_arr), label="E11_interp")
        canvas.axs.plot(t_arr, self.E2_func(t_arr), label="E2_interp")
        canvas.axs.plot(t_arr, self.E22_func(t_arr), label="E22_interp")
        canvas.axs.plot(t_arr, np.sqrt(self.E1_func(t_arr)**2+self.E11_func(t_arr)**2), label="E1+E11")
        canvas.axs.plot(t_arr, np.sqrt(self.E2_func(t_arr)**2+self.E22_func(t_arr)**2), label="E2+E22")
        
        canvas.axs.legend()
        
        canvas.draw()
        # plt.plot(t_arr, self.E1_func(t_arr), label="E1_interp")
        # plt.plot(t_arr, self.E11_func(t_arr), label="E11_interp")
        # plt.plot(t_arr, self.E2_func(t_arr), label="E2_interp")
        # plt.plot(t_arr, self.E22_func(t_arr), label="E22_interp")
        # plt.plot(t_arr, np.sqrt(self.E1_func(t_arr)**2+self.E11_func(t_arr)**2), label="E1+E11")
        # plt.plot(t_arr, np.sqrt(self.E2_func(t_arr)**2+self.E22_func(t_arr)**2), label="E2+E22")
        # plt.legend()
        # plt.show()
         
        
        
    def get_E(self, t):
        return self.E1_func(t), self.E11_func(t), self.E2_func(t), self.E22_func(t)


class DirectionCalculator:
    def __init__(self, q, phi_0_deg, phi_min_deg, phi_max_deg,
                 lambda_c, sll1_path, sll2_path, faz_path, approx_mode, freq_num, 
                 phi_pel_deg, K_n, phi_n_deg, noise_enable, poly_degree, adfr_canvas,
                   amp_canvas, e_canvas, phase_canvas, ampphase_canvas):
        # print(f"q={q}")
        # print(f"phi_0={phi_0_deg}")
        # print(f"phi_min={phi_min_deg}")
        # print(f"phi_max={phi_max_deg}")
        # print(f"lambda_c={lambda_c}")
        # print(f"sll1_path={sll1_path}")
        # print(f"sll2_path={sll2_path}")
        # print(f"faz_path={faz_path}")
        # print(f"approx_mode={approx_mode}") 
        # print(f"freq_num={freq_num}")
        # print(f"U1={U1}")
        # print(f"U2={U2}")
        # print(f"phi_pel={phi_pel_deg}")
        # print(f"K_n={K_n}")
        # print(f"phi_n={phi_n_deg}")
        # print(f"noise_enable={noise_enable}")
        # print(f"poly_degree={poly_degree}")
        # print(f"adfr_canvas={adfr_canvas}")
        # print(f"amp_canvas={amp_canvas}")
        # print(f"e_canvas={e_canvas}")
        # print(f"phase_canvas={phase_canvas}")
        # print(f"ampphase_canvas={ampphase_canvas}")
        self.phi_min, self.phi_max = (np.deg2rad(phi_min_deg),
                                      np.deg2rad(phi_max_deg))  # диапазон углов
        self.q = q  # сигнал/шум
        self.noise = Noise(q, noise_enable)  # Шум
        self.freq_num = freq_num
        self.ADFR = ADFR(sll1_path, sll2_path, freq_num + 1, self.phi_min, self.phi_max, approx_mode, poly_degree, adfr_canvas)  # Пеленгационная характеристика
        self.PDFR = PDFR(faz_path, freq_num + 1, approx_mode, poly_degree)  # Фазовая характеристика
        self.faz_path = faz_path
        self.phi_pel = np.deg2rad(phi_pel_deg)
        self.lambda_c = lambda_c
        self.omega_c = 2 * pi * C / lambda_c
        self.E = E(phi_pel_deg, K_n, phi_n_deg, self.noise, self.omega_c, phi_0_deg, self.ADFR, self.PDFR, e_canvas)
        self.amp_canvas = amp_canvas
        self.phase_canvas = phase_canvas
        self.ampphase_canvas = ampphase_canvas
        

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
            phi = self.ADFR.get_phi(20*math.log10((A1)/(A2)))
            phi_arr.append(phi)
                
        phi_func = my_math.get_approx_func(t_arr, phi_arr, my_math.ApproxMode.POLY, 0)
        phi_approxed = [phi_func(t) for t in t_arr]
        
        self.amp_canvas.axs[0].cla()
        self.amp_canvas.axs[1].cla()
        
        self.amp_canvas.axs[0].plot(t_arr, A1_arr, label="A1")
        self.amp_canvas.axs[0].plot(t_arr, A2_arr, label="A2")
        self.amp_canvas.axs[1].plot(t_arr, phi_arr, label="phi")
        self.amp_canvas.axs[1].plot(t_arr, phi_approxed, label="phi_approxed")
        
        self.amp_canvas.axs[0].legend()
        self.amp_canvas.axs[1].legend()
        
        self.amp_canvas.draw()
        
        # fig, axs = plt.subplots(1, 2)
        # axs[0].plot(t_arr, A1_arr, label="A1")
        # axs[0].plot(t_arr, A2_arr, label="A2")
        # axs[1].plot(t_arr, phi_arr, label="phi")
        # axs[1].plot(t_arr, phi_approxed, label="phi_approxed")
        
        # axs[0].legend()
        # axs[1].legend()
        # plt.show()
        
        return phi_func(t_arr[0])

    def phase_method(self):
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
        delta_phase_normed_arr = []
        for delta_phase in delta_phase_arr:
            if delta_phase > pi:
                delta_phase = -2 * pi + delta_phase
            if delta_phase < -pi:
                delta_phase = 2 * pi + delta_phase
            delta_phase_normed_arr.append(delta_phase)
        delta_phase_approxed_func = my_math.get_approx_func(t_arr, delta_phase_normed_arr, my_math.ApproxMode.POLY, 0)
        delta_phase_approxed_arr = [delta_phase_approxed_func(t) for t in t_arr]
        delta_phase = delta_phase_approxed_arr[0]
        
        self.phase_canvas.axs[0].cla()
        self.phase_canvas.axs[1].cla()
        
        self.phase_canvas.axs[0].plot(t_arr, phase1_arr, label="phase1")
        self.phase_canvas.axs[0].plot(t_arr, phase2_arr, label="phase2")
        self.phase_canvas.axs[0].plot(t_arr, delta_phase_arr, label="delta_phase")
        self.phase_canvas.axs[0].plot(t_arr, delta_phase_normed_arr, label="delta_phase_normed")
        self.phase_canvas.axs[0].plot(t_arr, delta_phase_approxed_arr, label="delta_phase_approxed")
        self.phase_canvas.axs[0].legend()
        
        # fig, axs =plt.subplots(1, 2)
        # axs[0].plot(t_arr, phase1_arr, label="phase1")
        # axs[0].plot(t_arr, phase2_arr, label="phase2")
        # axs[0].plot(t_arr, delta_phase_arr, label="delta_phase")
        # axs[0].plot(t_arr, delta_phase_normed_arr, label="delta_phase_normed")
        # axs[0].plot(t_arr, delta_phase_approxed_arr, label="delta_phase_approxed")
        # axs[0].legend()
        
        data = pd.read_csv(self.faz_path, sep='\\t', engine='python')
        data = data.apply(np.deg2rad)
        x_arr = data['X'].values
        faz_arr = data[data.columns[self.freq_num + 1]].values
        
        faz_approxed_arr = [self.PDFR.get_faz(x) for x in x_arr]
        faz_approxed_normed_arr = [self.PDFR.get_normed_faz(x) for x in x_arr]
        deg_x_arr = np.rad2deg(x_arr)
        
        self.phase_canvas.axs[1].plot(deg_x_arr, np.rad2deg(faz_arr), label="faz")
        self.phase_canvas.axs[1].plot(deg_x_arr, np.rad2deg(faz_approxed_arr), label="faz_approxed")
        self.phase_canvas.axs[1].plot(deg_x_arr, np.rad2deg(faz_approxed_normed_arr), label="faz_approxed_normed")
        self.phase_canvas.axs[1].plot(deg_x_arr, np.rad2deg([delta_phase for _ in x_arr]), label="delta_phase")
        self.phase_canvas.axs[1].legend()
        
        self.phase_canvas.draw()
        
        # axs[1].plot(deg_x_arr, np.rad2deg(faz_arr), label="faz")
        # axs[1].plot(deg_x_arr, np.rad2deg(faz_approxed_arr), label="faz_approxed")
        # axs[1].plot(deg_x_arr, np.rad2deg(faz_approxed_normed_arr), label="faz_approxed_normed")
        # axs[1].plot(deg_x_arr, np.rad2deg([delta_phase for _ in x_arr]), label="delta_phase")
        # axs[1].legend()
        # plt.show()
        
        dphi_min = self.PDFR.get_faz(self.phi_max)
        dphi_max = self.PDFR.get_faz(self.phi_min)
        inters = self.find_phase_intersections(delta_phase, dphi_min, dphi_max)
        
        i = 0
        deg_inters = np.rad2deg(inters)
                        
        return deg_inters

    def find_phase_intersections(self, delta_phi, dphi_min, dphi_max):
        f = interpolate.interp1d(self.PDFR.get_faz(np.linspace(self.phi_min, self.phi_max, 100000)), np.linspace(self.phi_min, self.phi_max, 100000))
        inters = [f(delta_phi)]

        n = 1
        while 1:
            stop_flag = True
            delta1 = delta_phi + 2 * pi * n
            delta2 = delta_phi - 2 * pi * n

            if dphi_min <= delta1 <= dphi_max:
                inters.append(f(delta1))
                stop_flag = False
            if dphi_min <= delta2 <= dphi_max:
                inters.append(f(delta2))
                stop_flag = False
            n += 1
            if stop_flag:
                break    
        return inters

    def ampphase_method(self, angle_amp, angles_phase):
        x_arr = np.linspace(self.phi_min, self.phi_max, 2)
        i = 0
        self.ampphase_canvas.axs.cla()
        for inter in angles_phase:
            self.ampphase_canvas.axs.plot(x_arr, [inter for _ in x_arr], label="phi"+str(i))
            i += 1
            
        self.ampphase_canvas.axs.legend()
        self.ampphase_canvas.axs.plot(x_arr, [angle_amp for _ in x_arr], label="phi_amp")
        
        self.ampphase_canvas.draw()
        
        best = None
        best_diff = None
        for phi in angles_phase:
            if best_diff is None or abs(phi - angle_amp) < best_diff:
                best = phi
                best_diff = abs(phi - angle_amp)
                continue
            break
        return best
    
    def calculate(self):
        angle_amp = self.amplitude_method()
        angles_phase = self.phase_method()
        angle = self.ampphase_method(angle_amp, angles_phase)
        accuracy = self.calculate_accuracy(np.deg2rad(angle))
        return angle, accuracy

    def calculate_accuracy(self, phi):
        return ((self.q ** 0.5) * (self.PDFR.get_normed_faz(self.phi_pel)) * math.cos(phi)) ** -1
