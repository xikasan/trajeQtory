# -*- coding: utf-8 -*-

import numpy as np
import xsim
import pandas as pd
import matplotlib.pyplot as plt
from openjij.sampler import SQASampler
from IPython.display import clear_output
from tqdm import tqdm_notebook, tqdm


# unknown target function generator
def fgen(nx):
    A = np.random.normal(size=(nx, nx))

    def func(x):
        ret = x @ A @ x.T
        if len(x.shape) == 1:
            return ret
        return np.diag(ret)

    return func


# random sampling
def xgen(nx, num=1):
    if num > 1:
        return np.random.choice([0, 1], (num, nx), replace=True)
    return np.random.choice([0, 1], nx, replace=True)


class BlackboxOptimizer:

    def __init__(self, model, obj, nx, notebook=False):
        self.model = model
        self.obj = obj
        self.nx = nx

        self.xs = None
        self.ys = None
        self.num_sampled = 0

        self.best = 1
        self.logger = xsim.Logger()
        self.record = None

        self.sampler = SQASampler()

        self._notebook = notebook

    def initial_sampling(self, num_sample):
        self.xs = xgen(self.nx, num_sample)
        self.ys = self.obj(self.xs)
        self.best = np.min(self.ys)
        self.num_sampled += num_sample

    def sample2(self):
        self.xs = xgen(self.nx)
        self.ys = self.obj(self.xs)
        self.num_sampled += 1

    def sample(self, num_reads=10):
        qubo = self.model.get_qubo()
        dataset = self.sampler.sample_qubo(qubo, num_reads=num_reads)
        sol = dataset.first
        x = np.array(list(sol.sample.values()))
        y = self.obj(x)
        self.xs = x
        self.ys = y
        self.num_sampled += 1
        return x, y

    def fit(self, num_repeat=None):
        num_do_fit = 1 if not hasattr(self.ys, "__len__") else len(self.ys)
        num_repeat = num_do_fit if num_repeat is None else num_repeat
        if self.model.REPEAT_FIT:
            loss = np.mean([self.model.fit(self.xs, self.ys) for _ in range(num_repeat)])
        else:
            loss = self.model.fit(self.xs, self.ys)
        return loss

    def initial_sample_and_fit(self, num_sample):
        self.initial_sampling(num_sample)
        loss = self.fit()

        self.logger.store(
            sample=self.num_sampled,
            loss=loss,
            y=np.min(self.ys),
            best=self.best
        ).flush()

    def optimize(self, max_sample, num_repeat=None, viz=False):
        if viz:
            plot_logger = dict(step=[], value=[], best=[])

        # main loop
        barfunc = tqdm if not self._notebook else tqdm_notebook

        for s in barfunc(range(self.num_sampled+1, max_sample+1)):
            x, y = self.sample()
            loss = self.fit(num_repeat=num_repeat)
            self.best = np.min([y, self.best])

            # logging
            self.logger.store(
                sample=self.num_sampled,
                loss=loss,
                y=y,
                best=self.best
            ).flush()
            if viz:
                plot_logger["step"].append(s)
                plot_logger["value"].append(y)
                plot_logger["best"].append(self.best)

            # visualize
            if viz:
                clear_output(True)
                plt.plot(plot_logger["step"], plot_logger["value"])
                plt.plot(plot_logger["step"], plot_logger["best"])
                plt.legend()
                plt.show()

        ret = xsim.Retriever(self.logger)
        self.record = pd.DataFrame(dict(
            sample=ret.sample(),
            loss=ret.loss(),
            best=ret.best(),
            y=ret.y()
        ))
        return self.record
