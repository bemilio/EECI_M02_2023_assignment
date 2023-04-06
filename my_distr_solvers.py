from tvopt import costs, sets, distributed_solvers
import numpy as np

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