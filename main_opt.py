import networkx as nx
import tvopt.distributed_solvers
from tvopt import networks as nw
from tvopt import distributed_solvers as ds
from tvopt import costs
import numpy as np
from mycosts import TwoExpCost
from my_distr_solvers import admm_fix, newton_raphson
from numpy import random
import matplotlib.pyplot as plt
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
    N_it = 1000
    adj_mat = nw.random_graph(N, 0.1)
    net = nw.Network(adj_mat)

    f_i = []
    for i in range(N):
        a = 0.2* random.rand()
        b = .2* random.rand()
        c = random.rand()
        d = random.rand()

        f_i.append(TwoExpCost(a,b,c,d))
    f=costs.SeparableCost(f_i)

    problem = {"f": f, "network": net}
    x_admm = np.zeros((1, N, N_it+1))
    x_dpg = np.zeros((1, N, N_it+1))
    x_NR = np.zeros((1, N, N_it+2))
    f_val_admm = np.zeros(N_it)
    f_val_dpg = np.zeros(N_it)
    f_val_NR = np.zeros(N_it)
    z = {}
    y_NR = np.zeros((1,N))
    s_NR = np.zeros((1, N))
    g_NR = np.zeros((1,N))
    h_NR = np.zeros((1, N))
    for i in range(net.N):
        for j in net.neighbors[i]:
            z[i, j] = 0
    for t in range(N_it):
        x_admm[..., t+1], z = admm_fix(problem, z_0=z,  penalty=1, rel=.5, num_iter=1)
        f_val_admm[t] = sum(f.function(x_admm[...,t]))
        x_dpg[..., t + 1] = tvopt.distributed_solvers.dpgm(problem, step=0.1, x_0=x_dpg[..., t], num_iter=1)
        f_val_dpg[t] = sum(f.function(x_dpg[..., t]))
        x_NR[..., t+1], y_NR[:], s_NR[:], g_NR[:], h_NR[:] = \
            newton_raphson(problem, step=0.02, x_0=x_NR[..., t], y_0=y_NR, s_0=s_NR, g_0=g_NR, h_0=h_NR, num_iter=1)
        f_val_NR[t] = sum(f.function(x_NR[..., t]))
        print("Iteration " + str(t))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.1))
    ax.plot(range(N_it), f_val_admm, label="ADMM")
    ax.set_ylabel(r'$f(x)$', fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.grid()
    ax.set_xlim(0, N_it)
    ax.plot(range(N_it), f_val_dpg, label = "DG")
    ax.set_ylabel(r'$f(x)$', fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.plot(range(N_it), f_val_NR, label = "NR")
    ax.set_ylabel(r'$f(x)$', fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.grid()
    ax.set_xlim(0, N_it)
    plt.legend()
    plt.show(block=False)
    print("stop")
