from enum import Enum
import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate
import math

from scipy.signal import butter, filtfilt, savgol_filter

pi = math.pi


class ApproxMode(Enum):
    POLY = 0
    LINE = 1
    INTERP = 2


def db_to_times(db):
    return 10 ** (db / 10.0)


def get_approx_func(x_arr, y_arr, mode, poly_degree=10):
    if mode == ApproxMode.POLY:
        return np.poly1d(np.polyfit(x_arr, y_arr, poly_degree))
    
    if mode == ApproxMode.INTERP:
        return scipy.interpolate.interp1d(x_arr, y_arr)
    
    if mode == ApproxMode.LINE:
        return np.poly1d(np.polyfit(x_arr, y_arr, 1))

    # if mode == ApproxMode.SIN:

    #     def mapping(x, a, b, c, d):
    #         return d * (np.sin(a * x + b)) + c

    #     coeffs, _ = curve_fit(mapping, x_arr, y_arr, maxfev=10000)
    #     a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    #     return lambda x: d * (np.sin(a * x + b)) + c

    # if mode == ApproxMode.EXP:

    #     def mapping(x, a, b, c, d):
    #         return a * (math.e ** (b * (x + d))) + c

    #     coeffs, _ = curve_fit(mapping, x_arr, y_arr)
    #     a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    #     return lambda x: a * (math.e ** (b * (x + d))) + c
    # if mode == ApproxMode.BUTTER:  # Баттеруорта
    #     normal_cutoff = 0.3
    #     b, a = butter(10, normal_cutoff, btype="low", analog=False)

    #     y_new = filtfilt(b, a, y_arr)
    #     return scipy.interpolate.interp1d(x_arr, y_new)

    # if mode == ApproxMode.SAVGOL:  # Савитского-Голея
    #     y_new = savgol_filter(y_arr, 30, 10)
    #     return scipy.interpolate.interp1d(x_arr, y_new)

def rad2deg_arr(arr):
    new_arr = []
    for el in arr:
        new_arr.append(np.rad2deg(el))
    return new_arr           
