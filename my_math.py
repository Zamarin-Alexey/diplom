from enum import Enum
import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate
import math

from scipy.signal import butter, filtfilt, savgol_filter

pi = math.pi


class FilterMode(Enum):
    SIN = 1
    EXP = 2
    LINE = 3
    POLY = 4
    FAZ = 5
    BUTTER = 6
    SAVGOL = 7


def db_to_times(db):
    return 10 ** (db / 10.0)


def get_filtered_func(x_arr, y_arr, mode, poly_degree=10):
    if mode == FilterMode.POLY:
        coeffs = np.polyfit(x_arr, y_arr, poly_degree)
        return np.poly1d(coeffs)

    if mode == FilterMode.SIN:

        def mapping(x, a, b, c, d):
            return d * (np.sin(a * x + b)) + c

        coeffs, _ = curve_fit(mapping, x_arr, y_arr, maxfev=10000)
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        print(a, b, c, d)
        return lambda x: d * (np.sin(a * x + b)) + c

    if mode == FilterMode.EXP:

        def mapping(x, a, b, c, d):
            return a * (math.e ** (b * (x + d))) + c

        coeffs, _ = curve_fit(mapping, x_arr, y_arr)
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        return lambda x: a * (math.e ** (b * (x + d))) + c
    if mode == FilterMode.BUTTER:  # Баттеруорта
        normal_cutoff = 0.3
        b, a = butter(25, normal_cutoff, btype="low", analog=False)

        y_new = filtfilt(b, a, y_arr)
        return scipy.interpolate.interp1d(x_arr, y_new)

    if mode == FilterMode.SAVGOL:  # Савитского-Голея
        y_new = savgol_filter(y_arr, 50, 10)
        return scipy.interpolate.interp1d(x_arr, y_new)


def find_phase_line_coeffs(x_arr, y_arr):
    shifted_y = np.unwrap(y_arr[20:-20])

    def mapping(x, k, b):
        return k * x + b

    coeffs, _ = curve_fit(mapping, x_arr[20:-20], shifted_y)
    k, b = coeffs[0], coeffs[1]
    return k, b


def find_phase_intersections(k, b, delta_phi, phi_min, phi_max):
    T = abs(-2 * pi / k)

    phi = (delta_phi - b) / k
    inters = []
    while 1:
        if phi < phi_min:
            phi += T
            continue
        break

    while 1:
        if phi > phi_max:
            break
        inters.append(phi)
        phi += T

    return inters
