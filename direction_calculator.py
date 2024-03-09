import math
from random import random

c = 299792458


def deg_to_rad(deg):
    rad = deg * math.pi / 180.0
    return rad


def rad_to_deg(rad):
    deg = rad / math.pi * 180
    return deg


class Noise:
    def __init__(self, q):
        self.q = q

    def get_noise(self, U1, U2):
        A1 = U1 / self._db_to_times(self.q)
        A2 = U2 / self._db_to_times(self.q)
        N1 = A1 * random() * 2 * math.pi
        N2 = A2 * random() * 2 * math.pi
        return N1, N2

    @staticmethod
    def _db_to_times(db):
        return 10 ** (db / 10.0)


class SLL:

    def __init__(self, width_deg):
        width_rad = deg_to_rad(width_deg)
        self.a = math.pi / (width_rad / 2)
        self.b = 1.39156 / self.a

    def get_sll(self, phi_pel_rad):
        G1 = abs(math.sin(self.a * (phi_pel_rad + self.b)) / (self.a * (phi_pel_rad + self.b)))
        G2 = abs(math.sin(self.a * (phi_pel_rad - self.b)) / (self.a * (phi_pel_rad - self.b)))
        return G1, G2


class DirectionCalculator:

    def __init__(self, q, width_deg, phi_0_rad, d, K, phi_min_deg, phi_max_deg, lambda_c):
        self.q = q  # сигнал/шум
        self.noise = Noise(q)  # Шум
        self.sll = SLL(width_deg)  # ДНА
        self.phi_0 = phi_0_rad  # начальная фаза
        self.d = d  # длина базы
        self.K = K  # пеленгационная чувствительность
        self.phi_min, self.phi_max = deg_to_rad(phi_min_deg), deg_to_rad(phi_max_deg)  # диапазон углов
        self.lambda_c = lambda_c
        self.omega_c = 2 * math.pi * c / lambda_c

    def get_E(self, U1, U2, phi_pel_deg, t, K_n, phi_n_deg):
        phi_n = deg_to_rad(phi_n_deg)
        phi_pel = deg_to_rad(phi_pel_deg)

        G1, G2 = self.sll.get_sll(phi_pel)
        N1, N2 = self.noise.get_noise(U1, U2)
        E1 = U1 * G1 * math.cos(self.omega_c * t + self.phi_0) + N1
        E11 = U1 * G1 * math.sin(self.omega_c * t + self.phi_0) + N1
        E2 = K_n * U2 * G2 * math.cos(self.omega_c * t + self.phi_0 + phi_n + 2
                                      * math.pi * self.d * math.sin(phi_pel) / self.lambda_c) + N2
        E22 = K_n * U2 * G2 * math.sin(self.omega_c * t + self.phi_0 + phi_n + 2
                                       * math.pi * self.d * math.sin(phi_pel) / self.lambda_c) + N2
        return E1, E11, E2, E22

    @staticmethod
    def get_amplitude_and_phase(Ex, Exx):
        A = math.sqrt((Ex ** 2) + (Exx ** 2))
        phi = math.atan2(Exx, Ex)
        return A, phi

    def amplitude_method(self, A1, A2):
        A = self.K * ((A1 - A2) / (A1 + A2))
        return A

    def phase_method(self, phi_1_rad, phi_2_rad):
        delta_phi = 2 * math.pi * ((phi_2_rad - phi_1_rad) % 1) - math.pi
        inters = set()
        n = 0
        while 1:
            stop_flag = False
            pos_val = (delta_phi + 2 * math.pi * n) * self.lambda_c / (2 * math.pi * self.d)
            neg_val = (delta_phi + 2 * math.pi * n) * self.lambda_c / (2 * math.pi * self.d)

            n += 1
            if pos_val <= 1 and math.asin(pos_val) < self.phi_max:
                stop_flag = True
                inters.add(math.asin(pos_val))
            if neg_val >= -1 and math.asin(pos_val) > self.phi_min:
                stop_flag = True
                inters.add(math.asin(neg_val))
            if stop_flag:
                break
        return inters

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
        accuracy = ((self.q ** 0.5) * (2 * math.pi * self.d / self.lambda_c) * math.cos(phi)) ** -1
        return accuracy
