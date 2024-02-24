import math
from random import random

c = 299792458
lambda_c = 9.7 * (10 ** -2)
omega_c = 2 * math.pi * c / lambda_c

def deg_to_rad(deg):
    rad = deg * math.pi / 180.0
    return rad

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

    def get_sll(self, phi_pel):
        G1 = abs(math.sin(self.a * phi_pel + self.b) / (self.a * (phi_pel + self.b)))
        G2 = abs(math.sin(self.a * phi_pel - self.b) / (self.a * (phi_pel - self.b)))
        return G1, G2


class DirectionCalculator:

    def __init__(self, q, width_deg, phi_0, d, K, phi_min, phi_max):
        self.q = q  # сигнал/шум
        self.noise = Noise(q)  # Шум
        self.sll = SLL(width_deg)  # ДНА
        self.phi_0 = deg_to_rad(phi_0)  # начальная фаза
        self.d = d  # длина базы
        self.K = K  # пеленгационная чувствительность
        self.phi_min, self.phi_max = deg_to_rad(phi_min), deg_to_rad(phi_max)  # диапазон углов

    def get_E(self, U1, U2, phi_pel, t, K_n, phi_n):
        G1, G2 = self.sll.get_sll(phi_pel)
        N1, N2 = self.noise.get_noise(U1, U2)
        E1 = U1 * G1 * math.cos(omega_c * t + self.phi_0) + N1
        E11 = U1 * G1 * math.sin(omega_c * t + self.phi_0) + N1
        E2 = K_n * U2 * G2 * math.cos(omega_c * t + self.phi_0 + phi_n + 2
                                      * math.pi * self.d * math.sin(phi_pel) / lambda_c) + N2
        E22 = K_n * U2 * G2 * math.sin(omega_c * t + self.phi_0 + phi_n + 2
                                       * math.pi * self.d * math.sin(phi_pel) / lambda_c) + N2
        return E1, E11, E2, E22

    @staticmethod
    def get_amplitude_and_phase(Ex, Exx):
        A = math.sqrt(Ex ** 2 + Exx ** 2)
        phi = math.atan(Exx / Ex)
        return A, phi

    def amplitude_method(self, A1, A2):
        A = self.K * ((A1 - A2) / (A1 + A2))
        return A

    def phase_method(self, phi_1, phi_2):
        delta_phi = phi_2 - phi_1
        inters = set()
        n = 0
        while 1:
            stop_flag = True
            pos_phi = math.asin((delta_phi + 2 * math.pi * n) * lambda_c / (2 * math.pi * self.d))
            neg_phi = math.asin((delta_phi - 2 * math.pi * n) * lambda_c / (2 * math.pi * self.d))
            if pos_phi < self.phi_max:
                stop_flag = False
                inters.add(pos_phi)
            if neg_phi > self.phi_min:
                stop_flag = False
                inters.add(pos_phi)
            if stop_flag:
                break
        return inters

    @staticmethod
    def choose_right_phi(phi_amp: float, phi_phase: set[float]) -> float:
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
        accuracy = ((self.q ** 0.5) * (2 * math.pi * self.d / lambda_c) * math.cos(phi)) ** -1
        return accuracy

