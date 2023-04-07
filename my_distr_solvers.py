from tvopt import costs, sets, distributed_solvers
import numpy as np
from numpy import random

def admm_fix(problem, penalty, rel, z_0=None, num_iter=100):
    r"""
    Distributed relaxed alternating direction method of multipliers (ADMM).

    This function implements the distributed ADMM, see [#]_ and references
    therein. The algorithm is characterized by the following updates

    .. math:: x_i^\ell = \operatorname{prox}_{f_i / (\rho d_i)}
                                    ([\pmb{A}^\top z^\ell]_i / (\rho d_i))

    .. math:: z_{ij}^{\ell+1} = (1-\alpha) z_{ij}^\ell - \alpha z_{ji}^\ell
                              + 2 \alpha \rho x_j^\ell

    for :math:`\ell = 0, 1, \ldots`, where :math:`d_i` is node :math:`i`'s
    degree, :math:`\rho` and :math:`\alpha` are the penalty and relaxation
    parameters, and :math:`\pmb{A}` is the arc incidence matrix. The algorithm
    is guaranteed to converge to the optimal solution.

    Parameters
    ----------
    problem : dict
        A dictionary containing the network describing the multi-agent system
        and the cost describing the problem.
    penalty : float
        The penalty parameter :math:`\rho` of the algorithm (convergence is
        guaranteed for any positive value).
    rel : float
        The relaxation parameter :math:`\alpha` of the algorithm (convergence
        is guaranteed for values in :math:`(0,1)`).
    z_0 : ndarray, optional
        The initial value of the edge variables. It is a dictionary which maps
        from edge to the corresponding variable.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    z : dictionary
        The edge variables after `num_iter` iterations.

    References
    ----------
    .. [#] N. Bastianello, R. Carli, L. Schenato, and M. Todescato,
           "Asynchronous Distributed Optimization over Lossy Networks via
           Relaxed ADMM: Stability and Linear Convergence," IEEE Transactions
           on Automatic Control.
    """

    # unpack problem data
    f, net = problem["f"], problem["network"]

    x = np.zeros(f.dom.shape)

    # parameters for local proximal evaluations
    penalty_i = [1 / (penalty * net.degrees[i]) for i in range(net.N)]

    # initialize arc variables
    z = {}
    for i in range(net.N):
        for j in net.neighbors[i]:
            if z_0 is None:
                z[i, j] = np.zeros(f.dom.shape[:-1])
            else:
                z[i, j] = z_0[i, j]

    for l in range(int(num_iter)):

        for i in range(net.N):

            # primal update
            x[..., i] = f.proximal(sum([z[i, j] for j in net.neighbors[i]]) * penalty_i[i], penalty=penalty_i[i], i=i)

            # communication step
            for j in net.neighbors[i]: net.send(i, j, -z[i, j] + 2 * penalty * x[..., i])

        # update auxiliary variables
        for i in range(net.N):
            for j in net.neighbors[i]:
                z[i, j] = (1 - rel) * z[i, j] + rel * net.receive(i, j)

    return x, z

def newton_raphson(problem, step, x_0=0, y_0=0, s_0=0, g_0=0, h_0=0, num_iter=100):

    # unpack problem data
    f, net = problem["f"], problem["network"]

    x = np.zeros(f.dom.shape)
    y = np.zeros(f.dom.shape)
    s = np.zeros(f.dom.shape)
    g_old = np.zeros(f.dom.shape)
    h_old = np.zeros(f.dom.shape)
    g = np.zeros(f.dom.shape)
    h = np.zeros(f.dom.shape)
    x[...] = x_0
    y[...] = y_0
    s[...] = s_0
    g_old[...] = g_0
    h_old[...] = h_0
    x_old = np.zeros(f.dom.shape)

    for l in range(int(num_iter)):
        for i in range(net.N):
            g[..., i] = f.hessian(x[...,i], i=i) * x[..., i] - f.gradient(x[...,i], i=i)
            h[..., i] = f.hessian(x[..., i], i=i)
        y[:] = distributed_solvers.average_consensus(net, y + g - g_old, num_iter=1)
        s[:] = distributed_solvers.average_consensus(net, s + h - h_old, num_iter=1)
        g_old[..., i] = f.hessian(x[..., i], i=i) * x[..., i] - f.gradient(x[..., i], i=i)
        h_old[..., i] = f.hessian(x[..., i], i=i)
        x_old[:] = x
        x[:] = (1-step) * x + step * np.divide(y,s) if np.linalg.norm(s) >0.0001 else x
    return x, y, s, g, h

def robust_newton_raphson(problem, step, x_0=0, y_0=0, s_0=1, g_0=0, h_0=1, g_old_0=0, h_old_0=1, \
                          sigma_y_0=0, sigma_s_0=0, rho_y_0=None, rho_s_0=None, num_iter=100, p_packet_loss=0.1):
    # unpack problem data
    f, net = problem["f"], problem["network"]
    weights = [1 / (1 + net.degrees[i]) for i in range(net.N)]

    x = np.zeros(f.dom.shape)
    y = np.zeros(f.dom.shape)
    s = np.zeros(f.dom.shape)
    g_old = np.zeros(f.dom.shape)
    h_old = np.zeros(f.dom.shape)
    g = np.zeros(f.dom.shape)
    h = np.zeros(f.dom.shape)
    sigma_y = np.zeros(f.dom.shape)
    sigma_s = np.zeros(f.dom.shape)
    x[...] = x_0
    y[...] = y_0
    s[...] = s_0
    g[...] = g_0
    h[...] = h_0
    g_old[...] = g_old_0
    h_old[...] = h_old_0
    sigma_y[...] = sigma_y_0
    sigma_s[...] = sigma_s_0
    # initialize arc variables
    rho_y = {}
    rho_s = {}
    for i in range(net.N):
        for j in net.neighbors[i]:
            if rho_y_0 is None:
                rho_y[i, j] = np.zeros(f.dom.shape[:-1])
            else:
                rho_y[i, j] = rho_y_0[i, j]
            if rho_s_0 is None:
                rho_s[i, j] = np.zeros(f.dom.shape[:-1])
            else:
                rho_s[i, j] = rho_s_0[i, j]

    for l in range(int(num_iter)):
        i = np.random.randint(0, net.N)
        y[..., i] = weights[i] * (y[..., i])
        s[..., i] = weights[i] * (s[..., i])
        sigma_y[..., i] = sigma_y[..., i] + y[..., i]
        sigma_s[..., i] = sigma_s[..., i] + s[..., i]
        for j in net.neighbors[i]:
            if random.binomial(1,p_packet_loss)==0:
                y[...,j] = sigma_y[..., i] - rho_y[i, j] + y[..., j]
                s[...,j] = sigma_s[..., i] - rho_s[i, j] + s[..., j]
                rho_y[i, j] = sigma_y[..., i]
                rho_s[i, j] = sigma_s[..., i]
                # update
                g_old[..., j] = g[..., j]
                h_old[..., j] = h[..., j]
                x[..., j] = (1 - step) * x[..., j] + step * y[..., j] / s[..., j]
                g[..., j] = f.hessian(x[..., j], i=j) * x[..., j] - f.gradient(x[..., j], i=j)
                h[..., j] = f.hessian(x[..., j], i=j)
                y[..., j] = y[..., j] + g[..., j] - g_old[..., j]
                s[..., j] = s[..., j] + h[..., j] - h_old[..., j]
    return x, y, s, g, h, g_old, h_old, sigma_y, sigma_s, rho_y, rho_s