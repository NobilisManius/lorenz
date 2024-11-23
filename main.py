import numpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


sigma   = 10
beta    = 1
rho     = 50.4


def lorenz_system(vector, t, sigma, beta, rho):
    '''

    :param vector:
    :param t:
    :param sigma:
    :param beta:
    :param rho:
    :return:
    '''
    x, y, z = vector

    der_vector = [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]
    return der_vector


zeroes_pos = [0.5, 1.4, 5.5]
time_points = np.linspace(0, 50, 10001)

positions = odeint(lorenz_system, zeroes_pos, time_points, args=(sigma, beta, rho))
x_sol, y_sol, z_sol = positions[:, 0], positions[:, 1], positions[:, 2]

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
l_plot, = ax.plot(x_sol, y_sol, z_sol)

def update(frame):
    lower_lim = max(0, frame - 100)

    x_cur = x_sol[lower_lim:frame + 1]
    y_cur = y_sol[lower_lim:frame + 1]
    z_cur = z_sol[lower_lim:frame + 1]

    ax.set_xlim(min(x_cur), max(x_cur))
    ax.set_ylim(min(y_cur), max(y_cur))
    ax.set_zlim(min(z_cur), max(z_cur))

    l_plot.set_data(x_cur, y_cur)
    l_plot.set_3d_properties(z_cur)

    return l_plot


animation = FuncAnimation(fig, update, frames=len(time_points), interval=25)
plt.show()