from wild_type_model import WildType
from wild_type_model_new_OD_each_step import WildTypeEachStep
from wild_type_model_new_OD_each_step_mass_update import WildTypeMassUpdate
from wild_type_model_input_discrete import WildTypeDiscrete
from wild_type_model_input_continuous import WildTypeContinuous
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from constants import HRS_TO_SECS, OD_TO_COUNT_CONC

GC_ODs_N = pd.read_csv("data/GC_ODs_N.csv")
Time = GC_ODs_N.loc[:,'Time'].astype(np.float64)

# log transform and fit
WT_a_log10 = np.log10(GC_ODs_N.loc[:, 'WT_a'])

# Taken from https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return y
model_parameters = pd.read_csv("Model_Parameters_CEM.csv",header=0).dropna(axis=0, how='any')
model_parameters_dict = {key: val for key, val in zip(model_parameters["Parameter Name"],model_parameters["Value"])}
p0 = [max(WT_a_log10), np.median(Time), 1, min(WT_a_log10)]  # this is an mandatory initial guess
popt, pcov = curve_fit(sigmoid, Time, WT_a_log10, p0, method='dogbox')

fit_fun_log10 = lambda t: sigmoid(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 10, num=int(1e3))
plt.scatter(Time, WT_a_log10)
plt.plot(t, fit_fun_log10(t))
plt.legend(['data', 'Sigmoid'], loc='upper right')
plt.title('log(OD) fit to sigmoid function')
plt.show()

# plot untransformed data spline
fit_fun = lambda t: 10**fit_fun_log10(t)
plt.scatter(Time, np.power(10,WT_a_log10))
plt.plot(t, fit_fun(t))
plt.title('log(OD) fit to sigmoid function transformed')
plt.legend(['data', 'Sigmoid'], loc='upper right')
plt.show()

# create model

# MCP geometry
radius_mcp = 7e-8
mcp_surface_area = model_parameters_dict["mcp_surface_area"]
mcp_volume =  model_parameters_dict["mcp_volume"]

# cell geometry
cell_surface_area = model_parameters_dict["cell_surface_area"]
cell_volume = model_parameters_dict["cell_volume"]

# external volume geometry
external_volume =model_parameters_dict["external_volume"]
wild_type_model_each_step_update = WildTypeMassUpdate(fit_fun, Time.iloc[-1], mcp_surface_area, mcp_volume,
                                     cell_surface_area, cell_volume, external_volume)
wild_type_model_each_step = WildTypeEachStep(fit_fun, Time.iloc[-1], mcp_surface_area, mcp_volume,
                                        cell_surface_area, cell_volume, external_volume)
wild_type_model_disc = WildTypeDiscrete(fit_fun, Time.iloc[-1], mcp_surface_area, mcp_volume,
                                         cell_surface_area, cell_volume, external_volume)
wild_type_model_cont = WildTypeContinuous(fit_fun, Time.iloc[-1], mcp_surface_area, mcp_volume,
                                           cell_surface_area, cell_volume, external_volume)
PermMCPPolar =model_parameters_dict["PermMCPPolar"]
PermMCPNonPolar = model_parameters_dict["PermMCPNonPolar"]

# initialize parameters
params = {'PermMCPPropanediol': PermMCPPolar,
            'PermMCPPropionaldehyde': PermMCPNonPolar,
            'PermMCPPropanol': PermMCPPolar,
            'PermMCPPropionyl': PermMCPNonPolar,
            'PermMCPPropionate': PermMCPPolar,
            'nmcps': model_parameters_dict["nmcps"],
            'PermCellPropanediol': model_parameters_dict["PermCellPropanediol"],
            'PermCellPropionaldehyde':  model_parameters_dict["PermCellPropionaldehyde"],
            'PermCellPropanol':  model_parameters_dict["PermCellPropanol"],
            'PermCellPropionyl':  model_parameters_dict["PermCellPropionyl"],
            'PermCellPropionate':  model_parameters_dict["PermCellPropionate"],
            'VmaxCDEf': model_parameters_dict["VmaxCDEf"],
            'KmCDEPropanediol':  model_parameters_dict["KmCDEPropanediol"],
            'VmaxPf':  model_parameters_dict["VmaxPf"],
            'KmPfPropionaldehyde':  model_parameters_dict["KmPfPropionaldehyde"],
            'VmaxPr':  model_parameters_dict["VmaxPr"],
            'KmPrPropionyl':  model_parameters_dict["KmPrPropionyl"],
            'VmaxQf':  model_parameters_dict["VmaxQf"],
            'KmQfPropionaldehyde':   model_parameters_dict["KmQfPropionaldehyde"],
            'VmaxQr':  model_parameters_dict["VmaxQr"],
            'KmQrPropanol':  model_parameters_dict["KmQrPropanol"],
            'VmaxLf':  model_parameters_dict["VmaxLf"],
            'KmLPropionyl':  model_parameters_dict["KmLPropionyl"]}

# initialize initial conditions
init_conds = {'PROPANEDIOL_MCP_INIT': 0,
              'PROPIONALDEHYDE_MCP_INIT': 0,
              'PROPANOL_MCP_INIT': 0,
              'PROPIONYL_MCP_INIT': 0,
              'PROPIONATE_MCP_INIT': 0,
              'PROPANEDIOL_CYTO_INIT': 0,
              'PROPIONALDEHYDE_CYTO_INIT': 0,
              'PROPANOL_CYTO_INIT': 0,
              'PROPIONYL_CYTO_INIT': 0,
              'PROPIONATE_CYTO_INIT': 0,
              'PROPANEDIOL_EXT_INIT': 50,
              'PROPIONALDEHYDE_EXT_INIT': 0,
              'PROPANOL_EXT_INIT': 0,
              'PROPIONYL_EXT_INIT': 0,
              'PROPIONATE_EXT_INIT': 0}

# run model for parameter set
time_each_step_update, sol_each_step_update = wild_type_model_each_step_update.generate_time_series(init_conds, params)
time_concat_each_step, sol_concat_each_step = wild_type_model_each_step.generate_time_series(init_conds, params)
time_concat_disc, sol_concat_disc = wild_type_model_disc.generate_time_series(init_conds, params)
time_concat_cont, sol_concat_cont = wild_type_model_cont.generate_time_series(init_conds, params)

names = ['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate']
c_test = ['-+','-+','-+','-+','-+']
c=['-*','-*','-*','-*','-*']
# plot MCP solutions
ymcp_each_step_update = sol_each_step_update[:, :5]
ymcp_each_step = sol_concat_each_step[:, :5]
ymcp_disc = sol_concat_disc[:, :5]
ymcp_cont = sol_concat_cont[:, :5]

for i in range(5):
    plt.plot(time_each_step_update/HRS_TO_SECS, ymcp_each_step_update[:,i],label="Model 1")
    plt.plot(time_concat_each_step/HRS_TO_SECS, ymcp_each_step[:,i],label="Model 2")
    plt.plot(time_concat_disc/HRS_TO_SECS, ymcp_disc[:,i],label="Model 3")
    plt.plot(time_concat_cont/HRS_TO_SECS, ymcp_cont[:,i],label="Model 4")

    plt.legend()
    plt.title('Plot of MCP ' +names[i]+ ' concentrations')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

# plot cellular solution
ycell_each_step_update = sol_each_step_update[:, 5:10]
ycell_each_step = sol_concat_each_step[:, 5:10]
ycell_disc = sol_concat_disc[:, 5:10]
ycell_cont = sol_concat_cont[:, 5:10]
for i in range(5):
    plt.plot(time_each_step_update/HRS_TO_SECS, ycell_each_step_update[:,i],label="Model 1")
    plt.plot(time_concat_each_step/HRS_TO_SECS, ycell_each_step[:,i],label="Model 2")
    plt.plot(time_concat_disc/HRS_TO_SECS, ycell_disc[:,i],label="Model 3")
    plt.plot(time_concat_cont/HRS_TO_SECS, ycell_cont[:,i],label="Model 4")
    plt.legend()
    plt.title('Plot of cytosol ' +names[i]+ ' concentrations')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

# plot external solution
yext_each_step_update = sol_each_step_update[:, 10:]
yext_each_step = sol_concat_each_step[:, 10:]
yext_disc = sol_concat_disc[:,  10:]
yext_cont = sol_concat_cont[:,  10:]
for i in range(5):
    plt.plot(time_each_step_update/HRS_TO_SECS, yext_each_step_update[:,i],label="Model 1")
    plt.plot(time_each_step_update/HRS_TO_SECS, yext_each_step[:,i],label="Model 2")
    plt.plot(time_concat_disc/HRS_TO_SECS, yext_disc[:,i],label="Model 3")
    plt.plot(time_concat_cont/HRS_TO_SECS, yext_cont[:,i],label="Model 4")
    plt.title('Plot of external ' +names[i]+ ' concentrations')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.legend()
    plt.show()

init_conds_list = np.array([val for val in init_conds.values()])

# conservation of mass formula
mcp_masses_org = init_conds_list[:5] * mcp_volume * params["nmcps"] * wild_type_model_each_step_update.optical_density_ts_disc[0]\
                 * OD_TO_COUNT_CONC * external_volume
cell_masses_org = init_conds_list[5:10] * cell_volume * wild_type_model_each_step_update.optical_density_ts_disc[0]* OD_TO_COUNT_CONC\
                  * external_volume
ext_masses_org = init_conds_list[10:] * external_volume

mcp_masses_fin = sol_each_step_update[-1,:5] * mcp_volume * params["nmcps"] * wild_type_model_each_step_update.optical_density_ts_disc[-1] \
                 * OD_TO_COUNT_CONC * external_volume
cell_masses_fin = sol_each_step_update[-1,5:10] * cell_volume * wild_type_model_each_step_update.optical_density_ts_disc[-1] * OD_TO_COUNT_CONC \
                  * external_volume
ext_masses_fin = sol_each_step_update[-1,10:] * external_volume

print("Original mass: " + str(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum()))
print("Final mass: " + str(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum()))
