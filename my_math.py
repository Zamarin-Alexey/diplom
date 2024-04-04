from enum import Enum
import numpy as np
from scipy.optimize import curve_fit
import math


class ApproxMode(Enum):
    SIN = 1
    EXP = 2
    LINE = 3
    POLY = 4


def deg_to_rad(deg):
    rad = deg * math.pi / 180.0
    return rad


def rad_to_deg(rad):
    deg = rad / math.pi * 180
    return deg


def db_to_times(db):
    return 10 ** (db / 10.0)


def get_approx_f(x_arr, y_arr, mode, poly_degree=10):
    if mode == ApproxMode.POLY:
        coeffs = np.polyfit(x_arr, y_arr, poly_degree)
        return np.poly1d(coeffs)

    if mode == ApproxMode.SIN:
        def mapping(x, a, b, c, d):
            return d * (np.sin(a * (deg_to_rad(x) + b))) + c

        coeffs, _ = curve_fit(mapping, x_arr, y_arr)
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        return lambda x: d * (np.sin(a * (deg_to_rad(x) + b))) + c

    if mode == ApproxMode.EXP:
        def mapping(x, a, b, c, d):
            return a * (math.e ** (b * (x + d))) + c

        coeffs, _ = curve_fit(mapping, x_arr, y_arr)
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        return lambda x: a * (math.e ** (b * (x + d))) + c
