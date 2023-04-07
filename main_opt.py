import networkx as nx
import tvopt.distributed_solvers
from tvopt import networks as nw
from tvopt import distributed_solvers as ds
from tvopt import costs
import numpy as np
from mycosts import TwoExpCost
from my_distr_solvers import admm_fix, newton_raphson, robust_newton_raphson
from numpy import random
import matplotlib.pyplot as plt
import scipy.optimize as opt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "font.serif": ["Computer Modern Roman"],
})
import os

if __name__ == '__main__':
    directory="."
    if not os.path.exists(directory + "/Figures"):
        os.makedirs(directory + r"/Figures")
    ###
    # Point 1: optimization
    ###
    random.seed(1)
    N=2
    N_it = 100
    adj_mat = nw.random_graph(N, 0.1)
    net = nw.Network(adj_mat)

    f_i = []
    a = []
    b = []
    c = []
    d = []
    for i in range(N):
        a.append(0.2* random.rand())
        b.append(.2* random.rand())
        c.append(random.rand())
        d.append(random.rand())

        f_i.append(TwoExpCost(a[i],b[i],c[i],d[i]))
    f=costs.SeparableCost(f_i)

    f_centralized = lambda x: sum([f_i[i].function(x) for i in range(N)])
    res = opt.minimize(f_centralized, 0)

    problem = {"f": f, "network": net}
    # Initialization
    #ADMM
    x_admm = np.zeros((N, N_it+1))
    z = {}
    for i in range(net.N):
        for j in net.neighbors[i]:
            z[i, j] = 0
    f_val_admm = np.zeros(N_it)
    # Grad
    x_dpg = np.zeros((N, N_it+1))
    f_val_dpg = np.zeros(N_it)
    # Newton-Raphson
    x_NR = np.zeros((N, N_it+1))
    y_NR = np.zeros((N))
    s_NR = np.zeros((N))
    g_NR = np.zeros((N))
    h_NR = np.zeros((N))
    f_val_NR = np.zeros(N_it)
    # Robust Newton-Raphson
    x_rNR = np.zeros((N, N_it + 1))
    y_rNR = np.zeros((N))
    s_rNR = np.ones((N))
    g_rNR = np.zeros((N))
    h_rNR = np.ones((N))
    g_old_rNR = np.zeros((N))
    h_old_rNR = np.ones((N))
    f_val_rNR = np.zeros(N_it)
    sigma_y_rNR = np.zeros((N))
    sigma_s_rNR = np.zeros((N))
    rho_y_rNR = {}
    for i in range(net.N):
        for j in net.neighbors[i]:
            rho_y_rNR[i, j] = 0
    rho_s_rNR = {}
    for i in range(net.N):
        for j in net.neighbors[i]:
            rho_s_rNR[i, j] = 0

    for t in range(N_it):
        # ADMM
        x_admm[..., t+1], z = admm_fix(problem, z_0=z,  penalty=.1, rel=.5, num_iter=1)
        f_val_admm[t] = sum(f.function(x_admm[...,t]))
        # Gradient
        x_dpg[..., t + 1] = tvopt.distributed_solvers.dpgm(problem, step=0.5, x_0=x_dpg[..., t], num_iter=1)
        f_val_dpg[t] = sum(f.function(x_dpg[..., t]))
        # Newton-Raphson
        x_NR[..., t+1], y_NR[:], s_NR[:], g_NR[:], h_NR[:] = \
            newton_raphson(problem, step=0.2, x_0=x_NR[..., t], y_0=y_NR, s_0=s_NR, g_0=g_NR, h_0=h_NR, num_iter=1)
        f_val_NR[t] = sum(f.function(x_NR[..., t]))
        # Robust Newton-Raphson
        x_rNR[..., t+1], y_rNR[:], s_rNR[:], g_rNR[:], h_rNR[:], \
        g_old_rNR[:], h_old_rNR[:], sigma_y_rNR[:], sigma_s_rNR[:], rho_y_rNR, rho_s_rNR = \
            robust_newton_raphson(problem, step=0.2, x_0=x_rNR[..., t], y_0=y_rNR, s_0=s_rNR, g_0=g_rNR, h_0=h_rNR,\
                                  g_old_0=g_old_rNR, h_old_0=h_old_rNR, sigma_y_0=sigma_y_rNR, sigma_s_0=sigma_s_rNR, \
                                  rho_y_0=rho_y_rNR, rho_s_0=rho_s_rNR, num_iter=1, p_packet_loss=0)
        f_val_rNR[t] = sum(f.function(x_rNR[..., t]))
        print("Iteration " + str(t))

    fig, ax = plt.subplots(2, 1, figsize=(6, 5.1), sharex=True)
    ax[0].plot(range(N_it), f_val_admm - res.fun, label="ADMM")
    ax[0].plot(range(N_it), f_val_dpg - res.fun, label = "DG")
    ax[0].plot(range(N_it), f_val_NR - res.fun, label = "NR")
    ax[0].plot(range(N_it), f_val_rNR - res.fun, label = "rob. NR")
    ax[0].set_ylabel(r'$f(x) - f^{\star}$', fontsize=9)
    ax[0].set_xlabel("Iteration", fontsize=9)
    ax[0].set_yscale('log')
    ax[0].set_ylim([10**(-3), 10**0])
    ax[0].grid()
    # ax[0].set_yscale('log')
    ax[0].set_xlim(0, N_it)
    ax[1].plot(range(N_it), np.linalg.norm(x_admm - np.average(x_admm, axis=0), axis=0)[:-1], label="ADMM")
    ax[1].plot(range(N_it), np.linalg.norm(x_dpg - np.average(x_dpg, axis=0), axis=0)[:-1], label = "DG")
    ax[1].plot(range(N_it), np.linalg.norm(x_NR - np.average(x_NR, axis=0), axis=0)[:-1], label = "NR")
    ax[1].plot(range(N_it), np.linalg.norm(x_rNR - np.average(x_rNR, axis=0), axis=0)[:-1], label = "rob. NR")
    ax[1].set_ylabel(r'$\|x - \frac{1}{N}\sum{x_i}\|$', fontsize=9)
    ax[1].set_xlabel("Iteration", fontsize=9)
    ax[1].set_yscale('log')
    ax[1].grid()
    plt.legend()
    plt.show(block=False)
    fig.savefig(directory + '/Figures/Distr_opt.png')
    fig.savefig(directory + '/Figures/Distr_opt.pdf')
    print("stop")
