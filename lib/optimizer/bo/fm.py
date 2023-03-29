# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def xmake(x):
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    if x.shape[0] > 1:
        xxs = [xmake(x_) for x_ in x]
        xxs = np.vstack(xxs)
        return xxs
    xx = x * x.T
    return xx.flatten()


class FactorizationMachine(nn.Module):

    REPEAT_FIT = False

    def __init__(self, nx, lr=1e-3, optimizer=optim.SGD):
        super().__init__()
        self.nx = nx
        self.l = nn.Linear(nx ** 2, 1, bias=False)
        self.lr = torch.tensor(lr)
        self.optim = optimizer(self.parameters(), lr=lr)

    def forward(self, xxs):
        return self.l(xxs)

    def predict(self, xs):
        if len(xs.shape) == 1:
            xs = xs.reshape((1, -1))
        xxs = xmake(xs)
        xxs = torch.from_numpy(xxs).float()
        ps = self(xxs)
        return ps.detach().numpy()

    def fit(self, xs, ys):
        if len(xs.shape) == 1:
            xs = xs.reshape((1, -1))
            ys = ys.reshape((1, -1))
        xxs = xmake(xs)
        xxs = torch.from_numpy(xxs).float()
        ys = torch.from_numpy(ys).float()
        ps = self(xxs)
        es = ps - ys
        loss = torch.mean(torch.square(es) * 0.5)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.detach().numpy()

    @property
    def weight(self):
        return self.l.weight.detach().numpy()

    def get_qubo(self):
        W = self.weight
        return W.reshape((self.nx, self.nx))
