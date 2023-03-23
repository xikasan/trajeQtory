# -*- coding: utf-8 -*-

import xsim
import numpy as np
import numpy.linalg as nl
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


# class IndexDrone2D:
#     x = 0
#     y = 1
#     u = 2
#     v = 3
#     pos = [x, y]
#     vel = [u, v]
#
#     cx = 0
#     cy = 1
#     c = [cx, cy]
#
#
# class Drone2D:
#     """Linear dynamics model of 2D multicopter
#     Motion is restricted in X-Y plane.
#     EoM:
#     dx = Ax + Bc
#     x = [x y u v]
#     c = [cx cy]
#     - - - - - - - - - - - - - - - - -
#     dx = u
#     dy = v
#     du = cx
#     """
#
#     ix = IndexDrone2D()
#
#     Cd = 0.05  # [Fs/m]
#     Vmax = 5.  # [m/s] maximum speed
#
#     R = 0.15  # [m] radius
#     m = 0.40  # [kg] mass
#
#     def __init__(self, dt: float = 0.01, dtype: np.dtype = np.float32, id_: int = 0):
#         self.dtype = dtype
#         self.dt: float = dt
#         self.id_: int = id_
#
#         self.x: np.ndarray = np.zeros(4)
#         self.A_, self.B_ = self.__build()
#         self.state_max: np.ndarray = np.array([100, 100, self.Vmax, self.Vmax], dtype=dtype)
#         self.pos_max: np.ndarray = np.array([100, 100], dtype=dtype)
#
#     def __build(self):
#         D = -1 * self.Cd
#         A = np.array([
#             [0, 0, 1, 0],
#             [0, 0, 0, 1],
#             [0, 0, D, 0],
#             [0, 0, 0, D]
#         ])
#         B = np.array([
#             [0, 0],
#             [0, 0],
#             [1, 0],
#             [0, 1]
#         ])
#         return A.T, B.T
#
#     def __call__(self, u: Union[List, np.ndarray]) -> np.ndarray:
#         u = np.asarray(u).astype(float)
#
#         def f(x):
#             return x @ self.A_ + u @ self.B_
#
#         dx = xsim.no_time_rungekutta(f, self.dt, self.x)
#         x = self.x + dx * self.dt
#
#         # position clipping
#         clipped_xy = np.clip(x[self.ix.pos], -self.pos_max, self.pos_max)
#         x[self.ix.pos] = clipped_xy
#         # velocity clipping
#         v = np.linalg.norm(x[self.ix.vel])
#         if v > self.Vmax:
#             vel = x[self.ix.vel] / v * self.Vmax
#             x[self.ix.vel] = vel
#         self.x = x
#         return self.state
#
#     def reset(self, xinit: np.ndarray = None) -> np.ndarray:
#         if xinit is not None:
#             xinit = np.asarray(xinit)
#             assert xinit.shape == self.x.shape,\
#                 f"state vector shape is expected {self.x.shape}, but {xinit.shape} is given."
#         else:
#             xinit = np.zeros_like(self.x)
#         self.x = xinit.astype(self.x.dtype)
#         return self.state
#
#     @property
#     def state(self) -> np.ndarray:
#         return self.x.astype(self.dtype)
#
#
# class DroneManager:
#
#     e: float = 0.9  # coefficient of restitution
#
#     def __init__(self, dt: float = 0.01, world_max: List[float] = [100, 100]):
#         self.dt = dt
#         self.drones: List[Drone2D] = []
#         self.world_max: List[float] = world_max
#
#     def add(self, drone: Drone2D):
#         assert isinstance(drone, Drone2D)
#         assert drone.dt == self.dt, \
#             "Simulation time step dt must same between drone and manager."
#         self.drones.append(drone)
#
#     def step(self, actions: List[List[float]]):
#         assert len(actions) == len(self.drones)
#         ix = self.drones[0].ix
#         pre_states = [drone.state for drone in self.drones]
#         states = [drone(act) for drone, act in zip(self.drones, actions)]
#         for ix1, drone1 in enumerate(self.drones):
#             pos1 = drone1.state[ix.pos]
#             vel1 = drone1.state[ix.vel]
#             for ix2, drone2 in enumerate(self.drones[ix1+1:]):
#                 pos2 = drone2.state[ix.pos]
#                 vel2 = drone2.state[ix.vel]
#
#                 p2 = pos2 - pos1
#                 d2 = nl.norm(p2)
#                 print(pos1, pos2, "|", d2, drone1.R + drone2.R, "||", vel1, vel2, "|!|!|", d2 > (drone1.R + drone2.R))
#
#                 if d2 > (drone1.R + drone2.R):
#                     continue
#
#                 nn = p2 / d2
#                 nt = np.array([nn[1], -nn[0]])
#
#                 vn1, vt1 = vel1 @ nn, vel1 @ nt
#                 vn2, vt2 = vel2 @ nn, vel2 @ nt
#
#                 vn1_ = (vn1 + vn2 + self.e * (vn2 - vn1)) / 2
#                 vn2_ = (vn1 + vn2 + self.e * (vn1 - vn2)) / 2
#
#                 vel1_ = vn1_ * nn + vt1 * nt
#                 vel2_ = vn2_ * nn + vt2 * nt
#
#                 drone1.x[ix.vel] = vel1_
#                 drone2.x[ix.vel] = vel2_
#
#     @property
#     def states(self):
#         return [
#             drone.state.copy() for drone in self.drones
#         ]
#
#     @property
#     def num_drones(self):
#         return len(self.drones)


from lib.dynamics.drone import Drone2D, IndexDrone2D, DroneRoll
from lib.dynamics.manager import DroneManager


def run3():
    manager = DroneManager()
    manager.create_drones(x=0., y= 0.05, u= 1.)
    manager.create_drones(x=1., y=-0.05, u=-1.)

    log = xsim.Logger()

    for time in xsim.generate_step_time(2, 0.01):
        states = manager.states
        actions = np.zeros((manager.num_drones, 2))

        if (time % 0.05) == 0:
            logdict_state = {f"state{i}": states[i] for i in range(manager.num_drones)}
            logdict_action = {f"action{i}": actions[i] for i in range(manager.num_drones)}
            log.store(time=time, **logdict_state, **logdict_action).flush()

        manager.step(actions)

    ix = IndexDrone2D()
    ret = xsim.Retriever(log)
    resdict = dict(
        time=ret.time(),
    )
    resdict.update(
        **{f"x{i}": getattr(ret, f"state{i}")(ix.x) for i in range(manager.num_drones)},
        **{f"y{i}": getattr(ret, f"state{i}")(ix.y) for i in range(manager.num_drones)},
        **{f"u{i}": getattr(ret, f"state{i}")(ix.u) for i in range(manager.num_drones)},
        **{f"v{i}": getattr(ret, f"state{i}")(ix.v) for i in range(manager.num_drones)},
    )
    res = pd.DataFrame(resdict)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    for i in range(manager.num_drones):
        res.plot(x=f"x{i}", y=f"y{i}", ax=axes[0], kind="scatter")
        res.plot(x=f"u{i}", y=f"v{i}", ax=axes[1], kind="scatter")

    axes[0].set_xlim(-0.5, 1.5)
    axes[0].set_ylim(-1, 1)
    plt.show()


def run2():
    manager = DroneManager()
    drone1 = Drone2D(id_=1)
    drone1.reset([0.0, 10.1,  1, 0])
    manager.add(drone1)

    drone2 = Drone2D(id_=2)
    drone2.reset([1.0, 10.0, -1., 0])
    manager.add(drone2)

    log = xsim.Logger()

    for time in xsim.generate_step_time(2, 0.01):
        states = manager.states
        actions = np.zeros((manager.num_drones, 2))

        if (time % 0.05) == 0:
            logdict_state = {f"state{i}": states[i] for i in range(manager.num_drones)}
            logdict_action = {f"action{i}": actions[i] for i in range(manager.num_drones)}
            log.store(time=time, **logdict_state, **logdict_action).flush()

        manager.step(actions)

    ix = drone1.ix
    ret = xsim.Retriever(log)
    resdict = dict(
        time=ret.time(),
    )
    resdict.update(
        **{f"x{i}": getattr(ret, f"state{i}")(ix.x) for i in range(manager.num_drones)},
        **{f"y{i}": getattr(ret, f"state{i}")(ix.y) for i in range(manager.num_drones)},
        **{f"u{i}": getattr(ret, f"state{i}")(ix.u) for i in range(manager.num_drones)},
        **{f"v{i}": getattr(ret, f"state{i}")(ix.v) for i in range(manager.num_drones)},
    )
    res = pd.DataFrame(resdict)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    for i in range(manager.num_drones):
        res.plot(x=f"x{i}", y=f"y{i}", ax=axes[0], kind="scatter")
        res.plot(x=f"u{i}", y=f"v{i}", ax=axes[1], kind="scatter")

    axes[0].set_xlim(-0.5, 1.5)
    axes[0].set_ylim(9, 11 )
    plt.show()


def run():
    drone = Drone2D()
    u = [1, 0]
    log = xsim.Logger()
    x = drone.reset()
    for time in xsim.generate_step_time(1, drone.dt):
        x = drone.state
        log.store(time=time, x=x, u=u).flush()

        drone(u)

    ix = drone.ix
    ret = xsim.Retriever(log)
    res = pd.DataFrame(dict(
        time=ret.time(),
        x=ret.x(ix.x),
        y=ret.x(ix.y),
        u=ret.x(ix.u),
        v=ret.x(ix.v),
    ))
    print(res)
    res.plot(x="time", y="x")
    plt.show()


def run4():
    roll = DroneRoll.striker
    print(roll is DroneRoll.striker)


if __name__ == '__main__':
    run4()
