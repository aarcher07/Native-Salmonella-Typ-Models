import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

y0 = [0]
fin_exp_time = 1
time_concat = []
sol_concat = []
n_discrete_tp = 10

time_discrete = np.linspace(0, 1, num=n_discrete_tp)
optical_density_ts_disc = []
f = lambda t: np.sin(t*2*10*np.pi)

for i in range(n_discrete_tp - 1):
    mean_OD = quad(f, time_discrete[i], time_discrete[i + 1])[0] / (time_discrete[i+1] - time_discrete[i])
    optical_density_ts_disc.append(mean_OD)

optical_density_ts_disc.append(mean_OD)
time_discrete2 = np.linspace(0, 1, num=1000)
plt.plot(time_discrete2,f(time_discrete2),label='$f(t)$')
plt.step(time_discrete, optical_density_ts_disc, where='post', label="step function")
plt.legend()
plt.show()
for i in range(n_discrete_tp - 1):

    #create OD problem
    ds = lambda t, x: optical_density_ts_disc[i]*np.sin(10*2*np.pi*x)+ 1

    #solve ODE
    sol = solve_ivp(ds, [time_discrete[i], time_discrete[i + 1]], y0, method="LSODA",
                    t_eval=np.linspace(time_discrete[i], time_discrete[i + 1], num=5, endpoint=True), atol=1e-6,
                    rtol=1e-6)
    # store
    time_concat = np.concatenate((time_concat, sol.t))
    sol_concat.extend(sol.y[0].T)
    #print(sol_concat)
    y0 = sol.y[:, -1].copy()

plt.plot(time_concat, sol_concat, label='Model 4')
ds = lambda t, x: f(t)*np.sin(10*2*np.pi*x) + 1
sol = solve_ivp(ds, [0, 1], [0], method="LSODA",
                t_eval=np.linspace(0,1, num=5*n_discrete_tp, endpoint=False), atol=1e-6,
                rtol=1e-6)
plt.plot(sol.t, sol.y.T, label='Model 5')

plt.legend()
plt.show()
