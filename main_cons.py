import networkx as nx
from tvopt import networks as nw
from tvopt import distributed_solvers as ds
import Consensus
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "font.serif": ["Computer Modern Roman"],
})
import os
import metr_hast_weights

if __name__ == '__main__':
    directory="."
    if not os.path.exists(directory + "/Figures"):
        os.makedirs(directory + r"/Figures")
    ###
    # Point 1: Ratio consensus
    ###
    random.seed(1)
    N=25
    connectivity = .3
    T = 30
    x_0 = 10*(random.rand(N) - .5)
    avg_x = np.mean(x_0)
    net = nx.fast_gnp_random_graph(N, connectivity, directed=True)
    while not nx.is_strongly_connected(net):
        net = nx.fast_gnp_random_graph(N, connectivity, directed=True)
    for i in net.nodes:
        # add self-loops
        net.add_edge(i,i)
    net = nx.stochastic_graph(net, weight="weight")
    # Standard ratio consensus
    alg_1 = Consensus.ratio_consensus(net, x_0)
    # Ratio consensus with modified initial y and s
    y_0 = 10*(random.rand(N) - .5) +1
    s_0 = 10*(random.rand(N) - .5) +1
    alg_2 = Consensus.ratio_consensus(net, x_0, y_0=y_0, s_0=s_0)
    x_1 = np.zeros((N, T))
    x_2 = np.zeros((N, T))
    x_1[:,0] = x_0[:]
    x_2[:,0] = x_0[:]
    for t in range(T-1):
        x_1[:,t+1] = alg_1.run_once()
        x_2[:,t+1] = alg_2.run_once()
    fig, ax = plt.subplots(2, 1, figsize=(6, 5.1))
    ax[0].plot(range(T), x_1.T - avg_x, label="Standard")
    ax[0].set_ylabel(r'$x$', fontsize=9)
    ax[0].set_xlabel("Iteration", fontsize=9)
    ax[0].grid()
    ax[0].set_xlim(0, T)
    ax[0].set_title("Ratio consensus")
    ax[1].plot(range(T), x_2.T - avg_x, label="Custom init. state")
    ax[1].set_ylabel(r'$x$', fontsize=9)
    ax[1].set_xlabel("Iteration", fontsize=9)
    ax[1].grid()
    ax[1].set_xlim(0, T)
    ax[1].set_ylim(-10, 10)
    ax[1].set_title("Ratio consensus with custom initial aux. var.")
    plt.tight_layout()
    fig.tight_layout()
    plt.draw()
    plt.show(block=False)
    fig.savefig(directory + '/Figures/Ratio_consensus.png')
    fig.savefig(directory + '/Figures/Ratio_consensus.pdf')

    ###
    # Point 2: asymmetric broadcast
    ###
    alg = Consensus.asym_broadcast_doublecomm(net, x_0)
    T = 150 # slower convergence
    x = np.zeros((N, T))
    x[:, 0] = x_0[:]
    for t in range(T-1):
        x[:,t+1] = alg.run_once()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.1))
    ax.plot(range(T), x.T - avg_x, label="Standard")
    ax.set_ylabel(r'$x$', fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.grid()
    ax.set_xlim(0, T)
    ax.set_title("Modified asym. broadcast")
    plt.show(block=False)
    fig.savefig(directory + '/Figures/mod_asym_broad.png')
    fig.savefig(directory + '/Figures/mod_asym_broad.pdf')

    ###
    # Point 3: ratio with noise
    ###
    alg = Consensus.avg_cons_with_noise(net, x_0, comm_noise_var = 1.)
    x = np.zeros((N, T))
    x[:, 0] = x_0
    for t in range(T - 1):
        x[:, t + 1] = alg.run_once()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.1))
    ax.plot(range(T), x.T - avg_x, label="Standard")
    ax.set_ylabel(r'$x$', fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.grid()
    ax.set_xlim(0, T)
    ax.set_ylim(-10, 10)
    ax.set_title("Ratio consensus with noise")
    plt.show(block=False)
    fig.savefig(directory + '/Figures/noisy_consensus.png')
    fig.savefig(directory + '/Figures/noisy_consensus.pdf')

    ###
    # Point 4: not connected graph
    ###
    nc_net = nx.fast_gnp_random_graph(N, 0.2, directed=True)
    while nx.is_strongly_connected(nc_net):
        nc_net = nx.fast_gnp_random_graph(N, 0.2, directed=True)
    for i in nc_net.nodes:
        # add self-loops
        nc_net.add_edge(i, i)
    nc_net = nx.stochastic_graph(nc_net)
    # Standard ratio consensus
    alg = Consensus.avg_cons_with_noise(nc_net, x_0, comm_noise_var=0)
    x = np.zeros((N, T))
    x[:, 0] = x_0
    for t in range(T-1):
        x[:,t+1] = alg.run_once()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.1))
    ax.plot(range(T), x.T, label="Standard")
    ax.set_ylabel(r'$x$', fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.grid()
    ax.set_xlim(0, T)
    ax.set_title("Ratio consensus with disconnected graph")
    plt.tight_layout()
    fig.tight_layout()
    plt.draw()
    plt.show(block=False)
    fig.savefig(directory + '/Figures/discon_graph.png')
    fig.savefig(directory + '/Figures/discon_graph.pdf')

    ###
    # Point 4: topologies_test
    ###
    complete = nx.complete_graph(N).to_directed()
    for i in net.nodes:
        # add self-loops
        complete.add_edge(i,i)
    complete = nx.stochastic_graph(complete)
    circle = nx.stochastic_graph(nx.cycle_graph(N).to_directed())
    for i in net.nodes:
        # add self-loops
        circle.add_edge(i,i)
    circle = nx.stochastic_graph(circle)
    star = nx.stochastic_graph(nx.star_graph(N-1).to_directed()) # -1 because networkx adds the star center
    for i in net.nodes:
        # add self-loops
        star.add_edge(i, i)
    star = nx.stochastic_graph(star)

    alg_1_complete = Consensus.avg_cons_with_noise(complete, x_0, comm_noise_var=0.)
    alg_1_circle = Consensus.avg_cons_with_noise(circle, x_0, comm_noise_var=0.)
    alg_1_star = Consensus.avg_cons_with_noise(star, x_0, comm_noise_var=0.)

    alg_2_complete = Consensus.robust_asynch_ratio(complete, x_0)
    alg_2_circle = Consensus.robust_asynch_ratio(circle, x_0)
    alg_2_star = Consensus.robust_asynch_ratio(star, x_0)

    x_1_complete = np.zeros((N, T))
    x_1_circle = np.zeros((N, T))
    x_1_star = np.zeros((N, T))
    x_1_complete[:, 0] = x_0
    x_1_circle[:, 0] = x_0
    x_1_star[:, 0] = x_0
    x_2_complete = np.zeros((N, T))
    x_2_circle = np.zeros((N, T))
    x_2_star = np.zeros((N, T))
    x_2_complete[:, 0] = x_0
    x_2_circle[:, 0] = x_0
    x_2_star[:, 0] = x_0
    for t in range(T - 1):
        x_1_complete[:, t + 1] = alg_1_complete.run_once()
        x_1_circle[:, t + 1] = alg_1_circle.run_once()
        x_1_star[:, t + 1] = alg_1_star.run_once()
        x_2_complete[:, t + 1] = alg_2_complete.run_once()
        x_2_circle[:, t + 1] = alg_2_circle.run_once()
        x_2_star[:, t + 1] = alg_2_star.run_once()

    fig, ax = plt.subplots(3, 2 , figsize=(6*2, 5.1*3))

    ax[0,0].plot(range(T), x_1_complete.T - avg_x, label="Standard")
    ax[0,0].set_ylabel(r'$x - \bar{x}$', fontsize=9)
    ax[0,0].set_xlabel("Iteration", fontsize=9)
    ax[0,0].grid()
    ax[0,0].set_xlim(0, T)
    ax[0,0].set_title("Avg. cons. - Complete")

    ax[1,0].plot(range(T), x_1_circle.T - avg_x, label="Standard")
    ax[1,0].set_ylabel(r'$x - \bar{x}$', fontsize=9)
    ax[1,0].set_xlabel("Iteration", fontsize=9)
    ax[1,0].grid()
    ax[1,0].set_xlim(0, T)
    ax[1,0].set_title("Avg. cons. - Circle")

    ax[2,0].plot(range(T), x_1_star.T - avg_x, label="Standard")
    ax[2,0].set_ylabel(r'$x - \bar{x}$', fontsize=9)
    ax[2,0].set_xlabel("Iteration", fontsize=9)
    ax[2,0].grid()
    ax[2,0].set_xlim(0, T)
    ax[2,0].set_title("Avg. cons. - Star")

    ax[0,1].plot(range(T), x_2_complete.T - avg_x, label="Standard")
    ax[0,1].set_ylabel(r'$x - \bar{x}$', fontsize=9)
    ax[0,1].set_xlabel("Iteration", fontsize=9)
    ax[0,1].grid()
    ax[0,1].set_xlim(0, T)
    ax[0,1].set_title("Robust ratio - Complete")

    ax[1,1].plot(range(T), x_2_circle.T - avg_x, label="Standard")
    ax[1,1].set_ylabel(r'$x - \bar{x}$', fontsize=9)
    ax[1,1].set_xlabel("Iteration", fontsize=9)
    ax[1,1].grid()
    ax[1,1].set_xlim(0, T)
    ax[1,1].set_title("Robust ratio - Circle")

    ax[2,1].plot(range(T), x_2_star.T - avg_x, label="Standard")
    ax[2,1].set_ylabel(r'$x - \bar{x}$', fontsize=9)
    ax[2,1].set_xlabel("Iteration", fontsize=9)
    ax[2,1].grid()
    ax[2,1].set_xlim(0, T)
    ax[2,1].set_title("Robust ratio - Star")

    plt.tight_layout()
    fig.tight_layout()
    plt.draw()
    plt.show(block=False)
    fig.savefig(directory + '/Figures/Compared_topologies.png')
    fig.savefig(directory + '/Figures/Compared_topologies.pdf')
