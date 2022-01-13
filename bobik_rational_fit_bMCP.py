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
OD_bMCPs = GC_ODs_N.loc[:,('Broken MCPs','OD')].astype(np.float64)
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

def rational3_3(x, p0, p1, p2, p3, q0,q1, q2):
    return rational(x, [p0, p1, p2, p3], [q0, q1, q2])

popt, pcov = curve_fit(rational3_3, Time, OD_bMCPs)

fit_fun = lambda t: rational3_3(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1], num=int(1e3))
plt.scatter(Time, OD_bMCPs)
plt.plot(t, fit_fun(t))
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD fit to sigmoid function')
plt.show()

def rational2_2(x, p0, p1, p2, q0,q1):
    return rational(x, [p0, p1, p2], [q0, q1])

popt, pcov = curve_fit(rational2_2, Time, OD_MCPs)

fit_fun = lambda t: rational2_2(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 2, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD fit to sigmoid function')
plt.show()

def rational2_2(x, p0, p1, q1):
    return rational(x, [p0, p1, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1])

popt, pcov = curve_fit(rational2_2, Time, OD_MCPs)

fit_fun = lambda t: rational2_2(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD fit to sigmoid function')
plt.show()

def rational3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1, q2])

popt, pcov = curve_fit(rational3_3, Time, OD_MCPs)

fit_fun = lambda t: rational3_3(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD fit to sigmoid function')
plt.show()

def rational4_4(x, p0, p1, p2,p3, q1, q2,q3):
    return rational(x, [p0, p1, p2,p3, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1, q2,q3])

popt, pcov = curve_fit(rational4_4, Time, OD_MCPs)

fit_fun = lambda t: rational4_4(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD fit to sigmoid function')
plt.show()

def rational5_5(x, p0, p1, p2,p3,p4, q1, q2,q3,q4):
    return rational(x, [p0, p1, p2,p3,p4, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1, q2,q3,q4])

popt, pcov = curve_fit(rational5_5, Time, OD_MCPs)

fit_fun = lambda t: rational5_5(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD fit to sigmoid function')
plt.show()


def rational6_6(x, p0, p1, p2,p3,p4,p5, q1, q2,q3,q4,q5):
    return rational(x, [p0, p1, p2,p3,p4,p5, OD_MCPs.iloc[0]], [p0/OD_MCPs.iloc[-1], q1, q2,q3,q4,q5])

popt, pcov = curve_fit(rational6_6, Time, OD_MCPs)

fit_fun = lambda t: rational6_6(t, *popt)
# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, OD_MCPs)
plt.plot(t, fit_fun(t))
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD fit to sigmoid function')
plt.show()
#
# # plot untransformed data spline
# fit_fun = lambda t: 10**fit_fun_log10(t)
# plt.scatter(Time, np.power(10,WT_a_log10))
# plt.plot(t, fit_fun(t))
# plt.title('log(OD) fit to sigmoid function transformed')
# plt.legend(['data', 'Sigmoid'], loc='upper right')
# plt.show()
#
# # create model
#
# # MCP geometry
# radius_mcp = 7e-8
# mcp_surface_area = 4*np.pi*(radius_mcp**2)
# mcp_volume = (4/3)*np.pi*(radius_mcp**3)
#
# # cell geometry
# cell_radius = 0.375e-6
# cell_length = 2.47e-6
# cell_surface_area = 2*np.pi*cell_radius*cell_length
# cell_volume = 4*np.pi/3*(cell_radius)**3 + np.pi*(cell_length - 2*cell_radius)*(cell_radius**2)
#
# # external volume geometry
# external_volume = 5e-5
#
# def sigmoid(x, L ,x0, k, b):
#     y = L / (1 + sp.exp(-k*(x-x0)))+b
#     return y
# fit_fun_log10 = lambda t: sigmoid(t, *popt)
# fit_fun = lambda t: 10**fit_fun_log10(t)
#
# wild_type_model = WildType(fit_fun, Time.iloc[-1], mcp_surface_area, mcp_volume,
#                            cell_surface_area, cell_volume, external_volume)
#
# PermMCPPolar =10 ** -2
# PermMCPNonPolar = 5 * 10 ** -3
#
# # initialize parameters
# params = {'PermMCPPropanediol': PermMCPPolar,
#             'PermMCPPropionaldehyde': PermMCPNonPolar,
#             'PermMCPPropanol': PermMCPPolar,
#             'PermMCPPropionyl': PermMCPNonPolar,
#             'PermMCPPropionate': PermMCPPolar,
#             'nmcps': 10,
#             'PermCellPropanediol': 10**-4,
#             'PermCellPropionaldehyde': 10**-2,
#             'PermCellPropanol': 10**-4,
#             'PermCellPropionyl': 10**-5,
#             'PermCellPropionate': 10**-7,
#             'VmaxCDEf': (3e2)*(1e2),
#             'KmCDEPropanediol': 0.5,
#             'VmaxPf': (3e2)*(1e2),
#             'KmPfPropionaldehyde': 0.5,
#             'VmaxPr': (3e2)*(1e2),
#             'KmPrPropionyl':  0.5,
#             'VmaxQf': (3e2)*(1e2),
#             'KmQfPropionaldehyde':  0.5,
#             'VmaxQr': (3e2)*(1e2),
#             'KmQrPropanol':  0.5,
#             'VmaxLf': (1e2),
#             'KmLPropionyl': 0.5}
#
# # initialize initial conditions
# init_conds = {'PROPANEDIOL_MCP_INIT': 0,
#               'PROPIONALDEHYDE_MCP_INIT': 0,
#               'PROPANOL_MCP_INIT': 0,
#               'PROPIONYL_MCP_INIT': 0,
#               'PROPIONATE_MCP_INIT': 0,
#               'PROPANEDIOL_CYTO_INIT': 0,
#               'PROPIONALDEHYDE_CYTO_INIT': 0,
#               'PROPANOL_CYTO_INIT': 0,
#               'PROPIONYL_CYTO_INIT': 0,
#               'PROPIONATE_CYTO_INIT': 0,
#               'PROPANEDIOL_EXT_INIT': 50,
#               'PROPIONALDEHYDE_EXT_INIT': 0,
#               'PROPANOL_EXT_INIT': 0,
#               'PROPIONYL_EXT_INIT': 0,
#               'PROPIONATE_EXT_INIT': 0}
#
# # run model for parameter set
# time_concat, sol_concat = wild_type_model.generate_time_series(init_conds, params)
#
# # plot MCP solutions
# yext = sol_concat[:, :5]
# plt.plot(time_concat/HRS_TO_SECS, yext)
# plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
# plt.title('Plot of MCP concentrations')
# plt.xlabel('time (hr)')
# plt.ylabel('concentration (mM)')
# plt.show()
#
# # plot cellular solution
# yext = sol_concat[:, 5:10]
# plt.plot(time_concat/HRS_TO_SECS, yext)
# plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
# plt.title('Plot of cytosol concentrations')
# plt.xlabel('time (hr)')
# plt.ylabel('concentration (mM)')
# plt.show()
#
# # plot external solution
# yext = sol_concat[:, 10:]
# plt.plot(time_concat/HRS_TO_SECS, yext)
# plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
# plt.title('Plot of external concentrations')
# plt.xlabel('time (hr)')
# plt.ylabel('concentration (mM)')
# plt.show()
#
# init_conds_list = np.array([val for val in init_conds.values()])
#
# # conservation of mass formula
# mcp_masses_org = init_conds_list[:5] * mcp_volume * params["nmcps"] * wild_type_model.optical_density_ts(Time.iloc[-1])\
#                  * OD_TO_COUNT_CONC * external_volume
# cell_masses_org = init_conds_list[5:10] * cell_volume * wild_type_model.optical_density_ts(Time.iloc[-1])* OD_TO_COUNT_CONC\
#                   * external_volume
# ext_masses_org = init_conds_list[10:] * external_volume
#
# mcp_masses_fin = sol_concat[-1,:5] * mcp_volume * params["nmcps"] * wild_type_model.optical_density_ts(Time.iloc[-1]) \
#                  * OD_TO_COUNT_CONC * external_volume
# cell_masses_fin = sol_concat[-1,5:10] * cell_volume * wild_type_model.optical_density_ts(Time.iloc[-1]) * OD_TO_COUNT_CONC \
#                   * external_volume
# ext_masses_fin = sol_concat[-1,10:] * external_volume
#
# print("Original mass: " + str(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum()))
# print("Final mass: " + str(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum()))
