# -*- coding: utf-8 -*-

import xsim
import numpy as np
import numpy.linalg as nl
from typing import List, Union
from enum import Enum, auto


class DroneRoll(Enum):
    attacker = auto()
    striker = auto()
    defense = auto()


class IndexDrone2D:
    x = 0
    y = 1
    u = 2
    v = 3
    pos = [x, y]
    vel = [u, v]

    cx = 0
    cy = 1
    c = [cx, cy]


class Drone2D:
    """Linear dynamics model of 2D multicopter
    Motion is restricted in X-Y plane.
    EoM:
    dx = Ax + Bc
    x = [x y u v]
    c = [cx cy]
    - - - - - - - - - - - - - - - - -
    dx = u
    dy = v
    du = cx
    """

    ix = IndexDrone2D()

    Cd = 0.05  # [Fs/m]
    Vmax = 5.  # [m/s] maximum speed

    R = 0.15  # [m] radius
    m = 0.40  # [kg] mass

    def __init__(self, roll: DroneRoll, dt: float = 0.01, dtype: np.dtype = np.float32, id_: int = 0):
        self.dtype = dtype
        self.dt: float = dt
        self.id_: int = id_
        self.roll: DroneRoll = roll

        self.x: np.ndarray = np.zeros(4)
        self.A_, self.B_ = self.__build()
        self.state_max: np.ndarray = np.array([10, 10, self.Vmax, self.Vmax], dtype=dtype)
        self.pos_max: np.ndarray = np.array([10, 10], dtype=dtype)
        self.act_max: np.ndarray = np.array([10, 10], dtype=dtype)

    def __build(self):
        D = -1 * self.Cd
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, D, 0],
            [0, 0, 0, D]
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])
        return A.T, B.T

    def __str__(self):
        return f"Drone2D: id={self.id_} roll={self.roll.name} pos={self.position} vel={self.velocity}"

    def __call__(self, u: Union[List, np.ndarray]) -> np.ndarray:
        u = np.asarray(u).astype(float)
        u = np.clip(u, -self.act_max, self.act_max)

        def f(x):
            return x @ self.A_ + u @ self.B_

        dx = xsim.no_time_rungekutta(f, self.dt, self.x)
        x = self.x + dx * self.dt

        # position clipping
        clipped_xy = np.clip(x[self.ix.pos], -self.pos_max, self.pos_max)
        x[self.ix.pos] = clipped_xy
        # velocity clipping
        v = np.linalg.norm(x[self.ix.vel])
        if v > self.Vmax:
            vel = x[self.ix.vel] / v * self.Vmax
            x[self.ix.vel] = vel
        self.x = x
        return self.state

    def reset(self, xinit: np.ndarray = None) -> np.ndarray:
        if xinit is not None:
            xinit = np.asarray(xinit)
            assert xinit.shape == self.x.shape,\
                f"state vector shape is expected {self.x.shape}, but {xinit.shape} is given."
        else:
            xinit = np.zeros_like(self.x)
        self.x = xinit.astype(self.x.dtype)
        return self.state

    @property
    def state(self) -> np.ndarray:
        return self.x.astype(self.dtype)

    @property
    def position(self) -> np.ndarray:
        return self.x[self.ix.pos].astype(self.dtype)

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.ix.vel].astype(self.dtype)
