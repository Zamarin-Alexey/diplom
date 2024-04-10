from enum import Enum
import numpy as np
from scipy.optimize import curve_fit
import math

pi = math.pi


class ApproxMode(Enum):
    SIN = 1
    EXP = 2
    LINE = 3
    POLY = 4
    FAZ = 5


def db_to_times(db):
    return 10 ** (db / 10.0)


def get_approx_f(x_arr, y_arr, mode, poly_degree=10):
    if mode == ApproxMode.POLY:
        coeffs = np.polyfit(x_arr, y_arr, poly_degree)
        return np.poly1d(coeffs)

    if mode == ApproxMode.SIN:
        def mapping(x, a, b, c, d):
            return d * (np.sin(a * (x + b))) + c

        coeffs, _ = curve_fit(mapping, x_arr, y_arr)
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        return lambda x: d * (np.sin(a * (x + b))) + c

    if mode == ApproxMode.EXP:
        def mapping(x, a, b, c, d):
            return a * (math.e ** (b * (x + d))) + c

        coeffs, _ = curve_fit(mapping, x_arr, y_arr)
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        return lambda x: a * (math.e ** (b * (x + d))) + c


def find_phase_line_coeffs(x_arr, y_arr):
    shifted_y = np.unwrap(y_arr)

    def mapping(x, k, b):
        return k * x + b

    coeffs, _ = curve_fit(mapping, x_arr, y_arr)
    k, b = coeffs[0], coeffs[1]
    return k, b


def find_phase_intersections(k, b, delta_phi, phi_min, phi_max):
    inters = set()
    n = 0
    while 1:
        stop_flag = True
        pos_val = (delta_phi + 2 * pi * n - b) / k
        neg_val = (delta_phi - 2 * pi * n - b) / k
        n += 1
        if pos_val <= phi_max:
            stop_flag = False
            inters.add(pos_val)
        if neg_val >= phi_min:
            stop_flag = False
            inters.add(neg_val)
        if stop_flag:
            break
    return inters

