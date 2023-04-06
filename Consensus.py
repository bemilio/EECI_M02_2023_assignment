import numpy as np
from numpy import random
import copy


class ratio_consensus:
    def __init__(self,net, x_0, y_0=None, s_0=None):
        self.x = copy.copy(x_0)
        if y_0 is not None:
            self.y = y_0
        else:
            self.y = x_0
        if s_0 is not None:
            self.s = s_0
        else:
            self.s = np.ones(len(net.nodes()))
        self.net = net

    def run_once(self):
        s = self.s
        y = self.y
        y_new = np.zeros(y.shape)
        s_new = np.zeros(s.shape)
        for i in self.net.nodes:
            y_new[i] = y[i]/(self.net.out_degree(i) + 1)
            s_new[i] = s[i]/(self.net.out_degree(i) + 1)
            for edge in self.net.in_edges(i):
                j = edge[0]
                y_new[i] = y_new[i] + y[j]/(self.net.out_degree(j) + 1)
                s_new[i] = s_new[i] + s[j]/(self.net.out_degree(j) + 1)
        self.y = y_new
        self.s = s_new
        self.x = np.divide(y_new,s_new)
        return self.x

class asym_broadcast_doublecomm:
    def __init__(self,net, x_0, stepsize = .5):
        self.x = copy.copy(x_0)
        self.net = net
        self.q = stepsize

    def run_once(self):
        N = len(self.net.nodes)
        i = np.random.choice(range(N), 2, replace=False)
        for j in self.net.neighbors(i[0]):
            if j not in self.net.neighbors(i[1]):
                self.x[j] = self.q * self.x[j] + (1-self.q) * self.x[i[0]]
            else:
                self.x[j] = self.q * self.x[j] + ((1 - self.q)/2) * self.x[i[0]] + ((1 - self.q)/2) * self.x[i[1]]
        for j in self.net.neighbors(i[1]):
            if j not in self.net.neighbors(i[1]):
                self.x[j] = self.q * self.x[j] + (1-self.q) * self.x[i]
        return self.x

class avg_cons_with_noise:
    def __init__(self,net, x_0, comm_noise_var =0):
        self.x = copy.copy(x_0)
        self.net = net
        self.comm_noise_var = comm_noise_var

    def run_once(self):
        x = self.x
        x_new = np.zeros(x.shape)
        for i in self.net.nodes():
            x_new[i] = x[i] * self.net.edges()[(i,i)]['weight']
            for j in self.net.neighbors(i):
                if not i==j:
                    w = self.comm_noise_var * random.randn()
                    x_new[i] = x_new[i] + (x[j] + w) * self.net.edges()[(i,j)]['weight']
        self.x = x_new
        return self.x

class robust_asynch_ratio:
    def __init__(self, net, x_0):
        self.x = copy.copy(x_0)
        self.y = x_0
        self.s = np.ones(len(net.nodes()))
        self.sigma_y = np.zeros(len(net.nodes()))
        self.sigma_s = np.zeros(len(net.nodes()))
        self.rho_y = np.zeros((len(net.nodes()), len(net.nodes())))
        self.rho_s = np.zeros((len(net.nodes()), len(net.nodes())))
        self.net = net

    def run_once(self):
        s = self.s
        y = self.y
        sigma_y = self.sigma_y
        sigma_s = self.sigma_s
        rho_y = self.rho_y
        rho_s = self.rho_s
        y_new = copy.copy(y)
        s_new = copy.copy(s)
        sigma_y_new = copy.copy(sigma_y)
        sigma_s_new = copy.copy(sigma_s)
        rho_y_new = copy.copy(rho_y)
        rho_s_new = copy.copy(rho_s)
        N = len(self.net.nodes)
        i = np.random.randint(0,N)
        y_new[i] = y[i]/(self.net.out_degree(i) + 1)
        s_new[i] = s[i]/(self.net.out_degree(i) + 1)
        sigma_y_new[i] = sigma_y[i] + y_new[i]
        sigma_s_new[i] = sigma_s[i] + s_new[i]
        for edge in self.net.out_edges(i):
            j = edge[1]
            y_new[j] = y_new[j] + sigma_y_new[i] - rho_y[j,i]
            s_new[j] = s_new[j] + sigma_s_new[i] - rho_s[j,i]
            rho_y_new[j, i] = sigma_y_new[i]
            rho_s_new[j, i] = sigma_s_new[i]
        self.y = y_new
        self.s = s_new
        self.sigma_y = sigma_y_new
        self.sigma_s = sigma_s_new
        self.rho_y = rho_y_new
        self.rho_s = rho_s_new
        self.x = np.divide(y_new, s_new)
        return self.x