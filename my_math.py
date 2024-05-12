from enum import Enum
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import math

from scipy.signal import butter, lfilter, savgol_filter

pi = math.pi


class ApproxMode(Enum):
    INTERP = 0
    CUBIC_SPLINE = 1
    LINE = 2
    POLY = 3
    CHEBYSHEV = 4
    LEGENDRE = 5
    LAGUERRE = 6
    HERMITE = 7
    BERNSTEIN = 8
    
class FilterAlgo(Enum):
    SAVGOL = 0
    BUTTER = 1
    
    
def db_to_times(db):
    return 10 ** (db / 10.0)


def get_approx_func(x_arr, y_arr, mode, poly_degree=10):
    if mode == ApproxMode.POLY:
        return np.poly1d(np.polyfit(x_arr, y_arr, poly_degree))
    
    if mode == ApproxMode.INTERP:
        return interpolate.interp1d(x_arr, y_arr)
    
    if mode == ApproxMode.LINE:
        return np.poly1d(np.polyfit(x_arr, y_arr, 1))
    
    if mode == ApproxMode.CUBIC_SPLINE:
        return interpolate.CubicSpline(x_arr, y_arr)
    
    if mode == ApproxMode.CHEBYSHEV:
        return np.polynomial.chebyshev.Chebyshev.fit(x_arr, y_arr, poly_degree)
    
    if mode == ApproxMode.HERMITE:
        return np.polynomial.hermite.Hermite.fit(x_arr, y_arr, poly_degree)
    
    if mode == ApproxMode.BERNSTEIN:
        return np.polynomial.polynomial.Polynomial.fit(x_arr, y_arr, poly_degree)
    
    if mode == ApproxMode.LEGENDRE:
        return np.polynomial.legendre.Legendre.fit(x_arr, y_arr, poly_degree)
    
    if mode == ApproxMode.LAGUERRE:
        return np.polynomial.laguerre.Laguerre.fit(x_arr, y_arr, poly_degree)
    
    
def get_filtered_func(x_arr, y_arr, algo, fs, cutoff_freq):   
    if algo == FilterAlgo.BUTTER:  # Баттеруорта
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        order = 5
        b, a = butter(order, normal_cutoff, btype="low", analog=False)

        y_new = lfilter(b, a, y_arr)
        return interpolate.interp1d(x_arr, y_new)

    if algo == FilterAlgo.SAVGOL:  # Савитского-Голея
        window = int(cutoff_freq/fs)+2
        polyorder = window // 2
        y_new = savgol_filter(y_arr, window, polyorder)
        return interpolate.interp1d(x_arr, y_new)
       
