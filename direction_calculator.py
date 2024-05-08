import math

from matplotlib import ticker
import numpy as np
import pandas as pd

import my_math

from scipy import interpolate
import matplotlib.pyplot as plt


C = 299792458
pi = math.pi
t_arr = np.linspace(0, 10**-9, 500)

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
    def __init__(self, sll1_path, sll2_path, col, phi_min_deg, phi_max_deg, approx_mode, poly_degree, canvas):
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
        self.G_func = my_math.get_approx_func(x_arr_trunc, pel_char_trunc, approx_mode, poly_degree)
        
        self.G1 = my_math.get_approx_func(x_arr_trunc, y1_arr[indices[0][0]: indices[0][-1] + 1], approx_mode, poly_degree)
        self.G2 = my_math.get_approx_func(x_arr_trunc, y2_arr[indices[0][0]: indices[0][-1] + 1], approx_mode, poly_degree)
        
        self.reversed_G_func = interpolate.interp1d(self.G_func(x_arr_trunc), x_arr_trunc, fill_value="extrapolate")
        
        approxed_pel = [self.G_func(x) for x in x_arr_trunc]
        
        canvas.figure.clear()
        ax = canvas.figure.subplots()
                
        ax.plot(x_arr, y1_arr, label="ДНА 1-й антенны")
        ax.plot(x_arr, y2_arr, label="ДНА 2-й антенны")
        ax.plot(x_arr, np.array(pel_char), label="Амплитудная пел. хар-ка")
        ax.plot(x_arr_trunc, np.array(approxed_pel), label="Аппроксимированная пел.хар-ка")
        ax.set_xlabel('φ, [град.]')
        ax.set_ylabel('A, [Вт]')
        ax.set_title("Амплитудная пеленгационная хар-ка")
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True)
        ax.legend(loc='upper right')
        canvas.draw()

        
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
    def __init__(self, phi_pel_deg, K_n, phi_n_deg, q, noise_enable,
                 omega_c, phi_0_deg, ADFR, PDFR, canvas, prefilter_en, prefilter_algo):
        phi_n_rad = np.deg2rad(phi_n_deg)
        phi_0_rad = np.deg2rad(phi_0_deg)
        
        B1 = ADFR.get_G1(phi_pel_deg)
        B2 = ADFR.get_G2(phi_pel_deg)
        U1 = 10 ** (B1/20)
        U2 = 10 ** (B2/20)
        
        noise = Noise(q, noise_enable)

        phi_pel_rad = np.deg2rad(phi_pel_deg)
        faz = PDFR.get_normed_faz(phi_pel_rad)
        
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
            self.E1_func = my_math.get_filtered_func(t_arr, E1, prefilter_algo)
            self.E11_func = my_math.get_filtered_func(t_arr, E11, prefilter_algo)
            self.E2_func = my_math.get_filtered_func(t_arr, E2, prefilter_algo)
            self.E22_func = my_math.get_filtered_func(t_arr, E22, prefilter_algo)
        else: 
            self.E1_func = interpolate.interp1d(t_arr, E1, kind='cubic')
            self.E11_func = interpolate.interp1d(t_arr, E11, kind='cubic')
            self.E2_func = interpolate.interp1d(t_arr, E2, kind='cubic')
            self.E22_func = interpolate.interp1d(t_arr, E22, kind='cubic')

        canvas.figure.clear()
        ax = canvas.figure.subplots()
        
        ax.plot(t_arr, self.E1_func(t_arr), label="E1")
        ax.plot(t_arr, self.E11_func(t_arr), label="E11")
        ax.plot(t_arr, self.E2_func(t_arr), label="E2")
        ax.plot(t_arr, self.E22_func(t_arr), label="E22")
        ax.set_xlabel('t, [с]')
        ax.set_ylabel('E, [В]')
        ax.set_title("Квадратуры на входе приемника")
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True)
        ax.legend(loc='upper right')
        
        canvas.draw()

         
    def get_E(self, t):
        return self.E1_func(t), self.E11_func(t), self.E2_func(t), self.E22_func(t)


class DirectionCalculator:
    def __init__(self, q, phi_0_deg, phi_min_deg, phi_max_deg,
                 lambda_c, sll1_path, sll2_path, faz_path, approx_mode, freq_num, 
                 phi_pel_deg, K_n, phi_n_deg, noise_enable, poly_degree, adfr_canvas,
                   amp_canvas, e_canvas, phase_canvas, ampphase_canvas, prefilter_en, prefilter_algo):
        self.phi_min, self.phi_max = (np.deg2rad(phi_min_deg),
                                      np.deg2rad(phi_max_deg))  # диапазон углов
        self.q = q  # сигнал/шум
        self.freq_num = freq_num
        self.ADFR = ADFR(sll1_path, sll2_path, freq_num + 1, phi_min_deg, phi_max_deg, approx_mode, poly_degree, adfr_canvas)  # Пеленгационная характеристика
        self.PDFR = PDFR(faz_path, freq_num + 1, approx_mode, poly_degree)  # Фазовая характеристика
        self.faz_path = faz_path
        self.phi_pel = np.deg2rad(phi_pel_deg)
        self.lambda_c = lambda_c
        self.omega_c = 2 * pi * C / lambda_c
        self.E = E(phi_pel_deg, K_n, phi_n_deg, q, noise_enable, self.omega_c, phi_0_deg, self.ADFR,
                   self.PDFR, e_canvas, prefilter_en, prefilter_algo)
        self.amp_canvas = amp_canvas
        self.phase_canvas = phase_canvas
        self.ampphase_canvas = ampphase_canvas
        

    def get_amplitude(self, Ex, Exx):
        return math.sqrt((Ex ** 2) + (Exx ** 2))

    def get_phase(self, Ex, Exx):
        return math.atan2(Exx, Ex)

    def amplitude_method(self):
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
        phi_abs_max = max(abs(phi) for phi in phi_arr)
        
        self.amp_canvas.figure.clear()
        axs = self.amp_canvas.figure.subplots(1, 2)
        
        minA1, maxA1 = min(A1_arr), max(A1_arr)
        minA2, maxA2 = min(A2_arr), max(A2_arr)
        mixAax = min(minA1, minA2, 0)
        maxAax = max(maxA1, maxA2, 0)
    
        axs[0].plot(t_arr, A1_arr, label="A1")
        axs[0].plot(t_arr, A2_arr, label="A2")
        axs[0].set_xlabel('t, [c]')
        axs[0].set_ylabel('A, [В]')
        axs[0].set_title("Амплитудный метод")
        axs[0].set_xlim(0, t_arr[-1])
        axs[0].set_ylim(mixAax, maxAax)
        axs[0].xaxis.set_major_locator(ticker.AutoLocator())
        axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0].grid(True)
        axs[0].legend(loc='upper right')
        
        axs[1].plot(t_arr, phi_arr, label="φ")
        axs[1].plot(t_arr, phi_approxed, label="φ_аппрокс.")
        axs[1].set_xlabel('t, [c]')
        axs[1].set_ylabel('φ, [град.]')
        axs[1].set_title("Фазовый метод")
        axs[1].set_xlim(0, t_arr[-1])
        axs[1].set_ylim(-phi_abs_max*1.1, phi_abs_max*1.1)
        axs[1].xaxis.set_major_locator(ticker.AutoLocator())
        axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1].grid(True)
        axs[1].legend(loc='upper right')
        
        self.amp_canvas.draw()
        
        return phi_func(t_arr[0])

    def phase_method(self):
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
        
        self.phase_canvas.figure.clear()
        axs = self.phase_canvas.figure.subplots(1, 2)
        
        axs[0].plot(t_arr, np.rad2deg(phase1_arr), label="Фаза1")
        axs[0].plot(t_arr, np.rad2deg(phase2_arr), label="Фаза2")
        axs[0].plot(t_arr, np.rad2deg(delta_phase_normed_arr), label="Разность фаз")
        axs[0].plot(t_arr, np.rad2deg(delta_phase_approxed_arr), label="Аппрокс. разность фаз")
        
        axs[0].set_xlabel('t, [с]')
        axs[0].set_ylabel('φ, [град.]')
        axs[0].set_title("Разность фаз")
        axs[0].xaxis.set_major_locator(ticker.AutoLocator())
        axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0].grid(True)
        axs[0].legend(loc='upper right')
        
        data = pd.read_csv(self.faz_path, sep='\\t', engine='python')
        data = data.apply(np.deg2rad)
        x_arr = data['X'].values
        faz_arr = data[data.columns[self.freq_num + 1]].values
        
        faz_approxed_arr = [self.PDFR.get_faz(x) for x in x_arr]
        faz_approxed_normed_arr = [self.PDFR.get_normed_faz(x) for x in x_arr]
        deg_x_arr = np.rad2deg(x_arr)
        
        axs[1].plot(deg_x_arr, np.rad2deg(faz_arr), label="Фазовая пел. хар-ка")
        axs[1].plot(deg_x_arr, np.rad2deg(faz_approxed_arr), label="Аппрокс. фаз. пел. хар-ка")
        axs[1].plot(deg_x_arr, np.rad2deg(faz_approxed_normed_arr), label="Норм. аппрокс. фаз. пел. хар-ка")
        axs[1].plot(deg_x_arr, np.rad2deg([delta_phase for _ in x_arr]), label="Разность фаз")
        axs[1].set_xlabel('Угол, [град.]')
        axs[1].set_ylabel('φ, [град.]')
        axs[1].set_title("Фазовая пел. хар-ка")
        axs[1].xaxis.set_major_locator(ticker.AutoLocator())
        axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1].grid(True)
        axs[1].legend(loc='upper right')
        
        self.phase_canvas.draw()
        
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
        x_arr = np.linspace(0, 10**-9, 2)
        i = 0
        
        self.ampphase_canvas.figure.clear()
        axs = self.ampphase_canvas.figure.subplots()
        
        best = None
        best_diff = None
        for phi in angles_phase:
            if best_diff is None or abs(phi - angle_amp) < best_diff:
                best = phi
                best_diff = abs(phi - angle_amp)

        for inter in angles_phase:
            width = 1
            if inter == best:
                width = 2
            axs.plot(x_arr, [inter for _ in x_arr], label="φ_фаз"+str(i+1), color='blue', linewidth=width)
            i += 1
            
        axs.plot(x_arr, [angle_amp for _ in x_arr], label="φ_амп", color='red')
        
        axs.set_xlabel('t, [с]')
        axs.set_ylabel('φ, [град.]')
        axs.set_title("Амплитудно-фазовый метод")
        axs.xaxis.set_major_locator(ticker.AutoLocator())
        axs.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs.grid(True)
        axs.legend(loc='upper right')
        
        axs.legend()
        self.ampphase_canvas.draw()
        
        
        return best
    
    def calculate(self):
        angle_amp = self.amplitude_method()
        angles_phase = self.phase_method()
        angle = self.ampphase_method(angle_amp, angles_phase)
        accuracy = self.calculate_accuracy(np.deg2rad(angle))
        return angle, accuracy

    def calculate_accuracy(self, phi):
        return (abs((self.q ** 0.5) * (self.PDFR.get_normed_faz(self.phi_pel)) * math.cos(phi)) ** -1)
