"""
Testing wild type model with a spline function trained on OD data.

Programme written by aarcher07
Editing History: See github history
"""

from wild_type_model import WildType
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from constants import HRS_TO_SECS, OD_TO_COUNT_CONC
import sympy as sp
GC_ODs_N = pd.read_excel("data/bobik_times_series_data_cleaned.xlsx", engine='openpyxl',header=[0,1]).dropna()

Time = GC_ODs_N.loc[:,('Time','Time (hrs)')].astype(np.float64)
OD_MCPs = GC_ODs_N.loc[:,('WT','OD')].astype(np.float64)

# log transform and fit

# Taken from https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
def rational(x, p, q):
    """
    The general rational function description.
    p is a list with the polynomial coefficients in the numerator
    q is a list with the polynomial coefficients (except the first one)
    in the denominator
    The zeroth order coefficient of the denominator polynomial is fixed at 1.
    Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
    zeroth order denominator coefficent must comes last. (Edited.)
    """
    return np.polyval(p, x) / np.polyval(q + [1.0], x)

def rational2_2(x, p0, p1, p2, q0,q1):
    return rational(x, [p0, p1, p2], [q0, q1])

popt, pcov = curve_fit(rational2_2, Time, OD_MCPs)

fit_fun = lambda t: rational2_2(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 2, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 2$')
plt.show()

def rational3_3(x, p0, p1, p2, p3, q0,q1, q2):
    return rational(x, [p0, p1, p2, p3], [q0, q1, q2])

popt, pcov = curve_fit(rational3_3, Time, OD_MCPs)

fit_fun = lambda t: rational3_3(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 2, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 3$')
plt.show()


def rational4_4(x, p0, p1, p2, p3,p4, q0,q1,q2,q3):
    return rational(x, [p0, p1, p2, p3,p4], [q0, q1,q2,q3])

popt, pcov = curve_fit(rational4_4, Time, OD_MCPs)

fit_fun = lambda t: rational4_4(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 2, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 4$')
plt.show()



def rational5_5(x, p0, p1, p2, p3,p4,p5, q0,q1, q2,q3,q4):
    return rational(x, [p0, p1, p2, p3,p4,p5], [q0, q1, q2,q3,q4])

popt, pcov = curve_fit(rational5_5, Time, OD_MCPs)

fit_fun = lambda t: rational5_5(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 1, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 5$')
plt.show()


def rational6_6(x, p0, p1, p2, p3,p4,p5, p6, q0,q1, q2,q3,q4,q5):
    return rational(x, [p0, p1, p2,p3,p4,p5,p6], [q0, q1,q2,q3,q4,q5])

popt, pcov = curve_fit(rational6_6, Time, OD_MCPs)

fit_fun = lambda t: rational6_6(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 2, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 6$')
plt.show()

########################################################################################################################
####### Constrained fits
########################################################################################################################

def rational2_2(x, p0, p1, q1):
    return rational(x, [p0, p1, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1])

popt, pcov = curve_fit(rational2_2, Time, OD_MCPs)

fit_fun = lambda t: rational2_2(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 2$')
plt.show()

def rational3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1, q2])

popt, pcov = curve_fit(rational3_3, Time, OD_MCPs)

fit_fun = lambda t: rational3_3(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 3$')
plt.show()

def rational4_4(x, p0, p1, p2,p3, q1, q2,q3):
    return rational(x, [p0, p1, p2,p3, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1, q2,q3])

popt, pcov = curve_fit(rational4_4, Time, OD_MCPs)

fit_fun = lambda t: rational4_4(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 1, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 4$')
plt.show()

def rational5_5(x, p0, p1, p2,p3,p4, q1, q2,q3,q4):
    return rational(x, [p0, p1, p2,p3,p4, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1, q2,q3,q4])

popt, pcov = curve_fit(rational5_5, Time, OD_MCPs)

fit_fun = lambda t: rational5_5(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 5$')
plt.show()


def rational6_6(x, p0, p1, p2,p3,p4,p5, q1, q2,q3,q4,q5):
    return rational(x, [p0, p1, p2,p3,p4,p5, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1, q2,q3,q4,q5])

popt, pcov = curve_fit(rational6_6, Time, OD_MCPs)

fit_fun = lambda t: rational6_6(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.xlabel('time')
plt.ylabel('OD')
plt.legend(['rational function', 'data'], loc='upper left')
plt.title('OD fit to rational function,' + r'$\frac{\sum_{i=0}^{n}p_ix^i}{1+\sum_{i=1}^{n}q_ix^i}$' + ', for $n = 6$')
plt.show()