from enum import Enum
import numpy as np
from scipy.optimize import curve_fit
import math


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
            return d * (np.sin(a * (np.deg2rad(x) + b))) + c

        coeffs, _ = curve_fit(mapping, x_arr, y_arr)
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        return lambda x: d * (np.sin(a * (np.deg2rad(x) + b))) + c

    if mode == ApproxMode.EXP:
        def mapping(x, a, b, c, d):
            return a * (math.e ** (b * (x + d))) + c

        coeffs, _ = curve_fit(mapping, x_arr, y_arr)
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        return lambda x: a * (math.e ** (b * (x + d))) + c
    
    if mode == ApproxMode.FAZ:
        shifted_y = np.unwrap([np.deg2rad(y) for y in y_arr])

        func = np.poly1d(np.polyfit(x_arr, shifted_y, 1))
        
        return lambda x: (np.rad2deg(func(x)) + 180) % 360 - 180


def find_intersections(func1, y, x_min, x_max):
    y_prev = func1(x_min)
    intersections = []

    for x in np.arange(x_min, x_max, 0.01):
        y_curr = func1(x)
        if y_curr > y and y_prev < y or y_curr < y and y_prev > y:
            intersections.append(x)
        y_prev = y_curr
