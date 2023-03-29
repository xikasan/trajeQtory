# -*- coding: utf-8 -*-

import numpy as np


def xmake(x):
    if len(x.shape) > 1:
        return np.vstack([xmake(x_) for x_ in x])
    d = len(x)
    p = 1 + d + d * (d - 1) // 2
    xvec = np.ones(p)
    xvec[1:d+1] = x
    xvec[d+1:] = [
        xi * xj
        for i, xi in enumerate(x) for j, xj in enumerate(x) if i > j
    ]
    return xvec.tolist()


class CombinatorialStructure():

    REPEAT_FIT = False

    def __init__(self, nx):
        self.d = d = nx
        self.p = p = 1 + d + d * (d - 1) // 2
        self.al = np.zeros(nx)

        # buffers
        self.X = []
        self.Y = []
        self.XX = []

        # BO parameters
        self.beta = np.random.standard_cauchy(p)
        self.tau = np.random.standard_cauchy(1)
        self.sigma = 0.1

    def fit(self, xs, ys):
        if len(xs.shape) == 1:
            xs = xs.reshape((1, -1))
            ys = ys.reshape((1, -1))
        xxs = xmake(xs)
        for i in range(xs.shape[0]):
            self.X.append(xs[i])
            self.Y.append(ys[i])
            self.XX.append(xxs[i])

        X = np.vstack(self.XX)
        y = np.vstack(self.Y)

        Sigma = self.tau ** 2 * np.diag(np.square(self.beta))
        A = X.T @ X + np.linalg.inv(Sigma)

        Ainv = np.linalg.inv(A)
        mu = Ainv @ X.T @ y
        var = self.sigma ** 2 * Ainv
        self.al = np.random.multivariate_normal(mu.flatten(), var)
        return np.mean(np.diag(var))

    def get_qubo(self):
        num_var = self.d
        Q = np.diag(self.al[1:num_var+1])
        counter = num_var + 1
        for i in range(num_var - 1):
            ix_de = counter
            ix_to = counter + num_var - i - 1
            Q[i, i+1:] = self.al[ix_de:ix_to]
            counter += ix_to - ix_de
        return Q
