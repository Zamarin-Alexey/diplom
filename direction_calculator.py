import math

from matplotlib import ticker
import numpy as np
import pandas as pd

import my_math

from scipy import interpolate
import time

pi = math.pi

class Noise:
    def __init__(self, q, enable):
        if enable:
            self.A = 1/math.sqrt(my_math.db_to_times(q))
        else:
            self.A = 0
        self.rng = np.random.default_rng()

    def get_noise(self):
        return self.A * (-1 + 2*self.rng.random())


class ADFR:
    def __init__(self, sll1_path, sll2_path, col, phi_min_deg, phi_max_deg, approx_mode, poly_degree):
        sll1_data = pd.read_csv(sll1_path, sep="\\t", engine="python")
        
        x_arr = sll1_data['X'].values

        y1_arr = sll1_data[sll1_data.columns[col]].values

        sll2_data = pd.read_csv(sll2_path, sep="\\t", engine="python")
        y2_arr = sll2_data[sll2_data.columns[col]].values
        pel_char = [y1_arr[i] - y2_arr[i] for i in range(len(y1_arr))]
        condition = (phi_min_deg < x_arr) & (x_arr < phi_max_deg)
        indices = np.where(condition)
        
        x_arr_trunc = x_arr[indices[0][0]: indices[0][-1] + 1]
        pel_char_trunc = pel_char[indices[0][0]: indices[0][-1] + 1]
        start = time.time()
        self.G_func = my_math.get_approx_func(x_arr_trunc, pel_char_trunc, approx_mode, poly_degree)
        self.G1 = my_math.get_approx_func(x_arr_trunc, y1_arr[indices[0][0]: indices[0][-1] + 1], approx_mode, poly_degree)
        self.G2 = my_math.get_approx_func(x_arr_trunc, y2_arr[indices[0][0]: indices[0][-1] + 1], approx_mode, poly_degree)
        end = time.time()
        
        self.reversed_G_func = interpolate.interp1d(self.G_func(x_arr_trunc), x_arr_trunc, fill_value="extrapolate")
        
        approxed_pel = [self.G_func(x) for x in x_arr_trunc]
        
        self.approx_time = end - start
        
        self.canvas_data = {}
        self.canvas_data["x_arr"] = x_arr
        self.canvas_data["y1_arr"] = y1_arr
        self.canvas_data["y2_arr"] = y2_arr
        self.canvas_data["pel_char"] = pel_char
        self.canvas_data["x_arr_trunc"] = x_arr_trunc
        self.canvas_data["approxed_pel"] = approxed_pel
        
    def get_canvas_data(self):
        return self.canvas_data

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
        start = time.time()
        self.faz_approx_func = my_math.get_approx_func(x_arr, shifted_faz_with_offset, approx_mode, poly_degree)
        end = time.time()
        self.approx_time = end - start
    
    def get_faz(self, x):
        return self.faz_approx_func(x)
    
    def get_normed_faz(self, x):
        return (self.faz_approx_func(x) - pi) % (-2 * pi) + pi
    

class E:
    def __init__(self, phi_pel_deg, K_n, phi_n_deg, q, noise_enable,
                 omega_c, phi_0_deg, ADFR, PDFR, prefilter_en, prefilter_algo, t_arr):
        phi_n_rad = np.deg2rad(phi_n_deg)
        phi_0_rad = np.deg2rad(phi_0_deg)
        
        B1 = ADFR.get_G1(phi_pel_deg)
        B2 = ADFR.get_G2(phi_pel_deg)
        U1 = 10 ** (B1/20)
        U2 = 10 ** (B2/20)
        
        noise = Noise(q, noise_enable)

        phi_pel_rad = np.deg2rad(phi_pel_deg)
        faz = PDFR.get_normed_faz(phi_pel_rad)
        
        fs = 1 / t_arr[1] * 1.2
        cutoff_freq = omega_c / (2 * pi)
        
        E1, E11, E2, E22 = [], [], [], []
        for t in t_arr:
            N1 = noise.get_noise()
            N11 = noise.get_noise()
            N2 = noise.get_noise()
            N22 = noise.get_noise()
            
            E1.append(U1 * (np.cos(omega_c * t + phi_0_rad) + N1))
            E11.append(U1 * (np.sin(omega_c * t + phi_0_rad) + N11))
            E2.append(K_n * U2 * (np.cos(omega_c * t + phi_0_rad + phi_n_rad + faz) + N2))
            E22.append(K_n * U2 * (np.sin(omega_c * t + phi_0_rad + phi_n_rad + faz) + N22))

        if prefilter_en:
            start = time.time()
            self.E1_func = my_math.get_filtered_func(t_arr, E1, prefilter_algo, fs, cutoff_freq)
            self.E11_func = my_math.get_filtered_func(t_arr, E11, prefilter_algo, fs, cutoff_freq)
            self.E2_func = my_math.get_filtered_func(t_arr, E2, prefilter_algo, fs, cutoff_freq)
            self.E22_func = my_math.get_filtered_func(t_arr, E22, prefilter_algo, fs, cutoff_freq)
            end = time.time()
            self.filter_time = end - start
        else: 
            start = time.time()
            self.E1_func = interpolate.interp1d(t_arr, E1, kind='cubic')
            self.E11_func = interpolate.interp1d(t_arr, E11, kind='cubic')
            self.E2_func = interpolate.interp1d(t_arr, E2, kind='cubic')
            self.E22_func = interpolate.interp1d(t_arr, E22, kind='cubic')
            end = time.time()
            self.filter_time = end - start

        self.canvas_data = {}
        self.canvas_data["t_arr"] = t_arr
        self.canvas_data["E1_func"] = self.E1_func
        self.canvas_data["E11_func"] = self.E11_func
        self.canvas_data["E2_func"] = self.E2_func
        self.canvas_data["E22_func"] = self.E22_func

    def get_canvas_data(self):
        return self.canvas_data

    def get_E(self, t):
        return self.E1_func(t), self.E11_func(t), self.E2_func(t), self.E22_func(t)


class DirectionCalculator:
    def __init__(self, q, phi_0_deg, phi_min_deg, phi_max_deg,
                 f_c, sll1_path, sll2_path, faz_path, approx_mode, freq_num, 
                 phi_pel_deg, K_n, phi_n_deg, noise_enable, poly_degree, prefilter_en, prefilter_algo, t, f_discr):
        self.phi_min, self.phi_max = (np.deg2rad(phi_min_deg),
                                      np.deg2rad(phi_max_deg))  # диапазон углов
        if noise_enable:
            self.q = q  # сигнал/шум
        else:
            self.q = float('inf')
        self.freq_num = freq_num
        self.ADFR = ADFR(sll1_path, sll2_path, freq_num + 1, phi_min_deg, phi_max_deg, approx_mode, poly_degree)  # Пеленгационная характеристика
        self.PDFR = PDFR(faz_path, freq_num + 1, approx_mode, poly_degree)  # Фазовая характеристика
        self.faz_path = faz_path
        self.phi_pel = np.deg2rad(phi_pel_deg)
        self.t_arr = np.arange(0, t, 1.0 / f_discr)
        omega_c = 2 * pi * f_c
        self.E = E(phi_pel_deg, K_n, phi_n_deg, q, noise_enable, omega_c, phi_0_deg, self.ADFR,
                   self.PDFR, prefilter_en, prefilter_algo, self.t_arr)
        
        self.phase_canvas_data = {}
        self.amp_canvas_data = {}
        self.ampphase_canvas_data = {}
        

    def get_amplitude(self, Ex, Exx):
        return math.sqrt((Ex ** 2) + (Exx ** 2))

    def get_phase(self, Ex, Exx):
        return math.atan2(Exx, Ex)

    def amplitude_method(self):
        A1_arr, A2_arr = [], []
        phi_arr = []
        for t in self.t_arr:
            E1, E11, E2, E22 = self.E.get_E(t)
            A1 = self.get_amplitude(E1, E11)
            A2 = self.get_amplitude(E2, E22)
            A1_arr.append(A1)
            A2_arr.append(A2)
            phi = self.ADFR.get_phi(20*math.log10((A1)/(A2)))
            phi_arr.append(phi)
                
        phi = np.average(phi_arr)
        phi_approxed = [phi for t in self.t_arr]
        
        self.amp_canvas_data["t_arr"] = self.t_arr
        self.amp_canvas_data["A1_arr"] = A1_arr
        self.amp_canvas_data["A2_arr"] = A2_arr
        self.amp_canvas_data["phi_approxed"] = phi_approxed
        self.amp_canvas_data["phi_arr"] = phi_arr

        return phi
    
        
    def get_amp_canvas_data(self):
        return self.amp_canvas_data
    

    def phase_method(self):
        phase1_arr, phase2_arr = [], []
        delta_phase_arr = []
        for t in self.t_arr:
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
        delta_phase = np.average(delta_phase_normed_arr)
        delta_phase_approxed_arr = [delta_phase for _ in self.t_arr]
        delta_phase = delta_phase_approxed_arr[0]
        
        data = pd.read_csv(self.faz_path, sep='\\t', engine='python')
        data = data.apply(np.deg2rad)
        x_arr = data['X'].values
        faz_arr = data[data.columns[self.freq_num + 1]].values
        
        faz_approxed_arr = [self.PDFR.get_faz(x) for x in x_arr]
        faz_approxed_normed_arr = [self.PDFR.get_normed_faz(x) for x in x_arr]
        deg_x_arr = np.rad2deg(x_arr)
        
        dphi_min = self.PDFR.get_faz(self.phi_max)
        dphi_max = self.PDFR.get_faz(self.phi_min)
        inters = self.find_phase_intersections(delta_phase, dphi_min, dphi_max)
        
        deg_inters = np.rad2deg(inters)

        self.phase_canvas_data["t_arr"] = self.t_arr
        self.phase_canvas_data["phase1_arr"] = phase1_arr
        self.phase_canvas_data["phase2_arr"] = phase2_arr
        self.phase_canvas_data["delta_phase_normed_arr"] = delta_phase_normed_arr
        self.phase_canvas_data["delta_phase_approxed_arr"] = delta_phase_approxed_arr
        self.phase_canvas_data["deg_x_arr"] = deg_x_arr
        self.phase_canvas_data["faz_arr"] = faz_arr
        self.phase_canvas_data["faz_approxed_arr"] = faz_approxed_arr
        self.phase_canvas_data["faz_approxed_normed_arr"] = faz_approxed_normed_arr
        self.phase_canvas_data["delta_phase"] = delta_phase
             
        return deg_inters
    
    def get_phase_canvas_data(self):
        return self.phase_canvas_data
        

    def find_phase_intersections(self, delta_phi, dphi_min, dphi_max):
        f = interpolate.interp1d(self.PDFR.get_faz(np.linspace(self.phi_min, self.phi_max, 1000)), np.linspace(self.phi_min, self.phi_max, 1000), fill_value="extrapolate")
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
        x_arr = np.linspace(0, self.t_arr[-1], 2)
        
        best = None
        best_diff = None
        for phi in angles_phase:
            if best_diff is None or abs(phi - angle_amp) < best_diff:
                best = phi
                best_diff = abs(phi - angle_amp)

        self.ampphase_canvas_data["angles_phase"] = angles_phase
        self.ampphase_canvas_data["best"] = best
        self.ampphase_canvas_data["x_arr"] = x_arr
        self.ampphase_canvas_data["angle_amp"] = angle_amp
        
        return best
    
    def get_ampphase_canvas_data(self):
        return self.ampphase_canvas_data
    
    def calculate(self):
        angle_amp = self.amplitude_method()
        angles_phase = self.phase_method()
        angle = self.ampphase_method(angle_amp, angles_phase)
        accuracy = self.calculate_accuracy(np.deg2rad(angle))
        return angle, accuracy

    def calculate_accuracy(self, phi):
        return (abs((self.q ** 0.5) * (self.PDFR.get_normed_faz(self.phi_pel)) * math.cos(phi)) ** -1)
