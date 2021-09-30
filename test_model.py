import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

y_init = [1]#[50,0]
y0 = y_init.copy()
fin_exp_time = 1
time_concat = []
sol_concat = []
n_discrete_tp = 1000

time_discrete = np.linspace(0, fin_exp_time, num=n_discrete_tp)
optical_density_ts_disc = []
f = lambda t: t
name = 'linear_time_dependence_flow_across_membrane_time_pts_'+ str(n_discrete_tp)

for i in range(n_discrete_tp - 1):
    mean_OD = quad(f, time_discrete[i], time_discrete[i + 1])[0] / (time_discrete[i+1] - time_discrete[i])
    optical_density_ts_disc.append(mean_OD)

optical_density_ts_disc.append(mean_OD)
time_discrete2 = np.linspace(0, fin_exp_time, num=1000)
plt.plot(time_discrete2,f(time_discrete2),label='$f(t)$')
plt.step(time_discrete, optical_density_ts_disc, where='post', label="discretized f(t)")
plt.legend(loc='upper right')
plt.savefig('figures/' + name + '_discretized_plot.png', bbox_inches='tight')
plt.close()
for i in range(n_discrete_tp - 1):

    #create OD problem
    ds = lambda t, x: np.array([optical_density_ts_disc[i]*1e8*(x[1]-x[0]),(x[0]-x[1])])

    #solve ODE
    sol = solve_ivp(ds, [time_discrete[i], time_discrete[i + 1]], y0, method="LSODA",
                    t_eval=np.linspace(time_discrete[i], time_discrete[i + 1], num=5, endpoint=True), atol=1e-6,
                    rtol=1e-6)
    # store
    time_concat = np.concatenate((time_concat, sol.t))
    sol_concat.extend(sol.y.T)
    #print(sol_concat)
    y0 = sol.y[:, -1].copy()
sol_concat = np.array(sol_concat).T

ds = lambda t, x: np.array([f(t)*1e8*(x[1]-x[0]),(x[0]-x[1])])
sol = solve_ivp(ds, [0, fin_exp_time], y_init, method="LSODA",
                t_eval=np.linspace(0,fin_exp_time, num=5*n_discrete_tp, endpoint=False), atol=1e-6,
                rtol=1e-6)

for i in range(sol_concat.shape[0]):
    plt.plot(time_concat, sol_concat[i,:], label='discretized f(t) solution')
    plt.plot(sol.t, sol.y[i,:], label='continuous f(t) solution')
    plt.legend(loc='upper right')
    plt.savefig('figures/' + name + '_sol' + str(i) + '_model_comparison.png', bbox_inches='tight')
    plt.close()
