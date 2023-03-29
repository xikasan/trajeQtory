# -*- coding: utf-8 -*-

import xsim
import numpy as np
import numpy.linalg as nl
from typing import List, Union

from .drone import Drone2D, IndexDrone2D, DroneRoll


class DroneManager:

    ix = IndexDrone2D()
    e: float = 0.9  # coefficient of restitution

    def __init__(self, dt: float = 0.01, world_max: List[float] = [10, 10]):
        self.dt = dt
        self.drones: List[Drone2D] = []
        self.world_max: List[float] = world_max

    def add(self, drone: Drone2D):
        assert isinstance(drone, Drone2D)
        assert drone.dt == self.dt, \
            "Simulation time step dt must same between drone and manager."
        self.drones.append(drone)

    def step(self, actions: Union[List[List[float]], np.ndarray]):
        assert len(actions) == len(self.drones)
        ix = self.ix
        states = [drone(act) for drone, act in zip(self.drones, actions)]
        for ix1, drone1 in enumerate(self.drones):
            pos1 = drone1.state[ix.pos]
            vel1 = drone1.state[ix.vel]
            for ix2, drone2 in enumerate(self.drones[ix1+1:]):
                pos2 = drone2.state[ix.pos]
                vel2 = drone2.state[ix.vel]

                p2 = pos2 - pos1
                d2 = nl.norm(p2)

                # collision check
                if d2 > (drone1.R + drone2.R):
                    continue

                nn = p2 / d2
                nt = np.array([nn[1], -nn[0]])

                vn1, vt1 = vel1 @ nn, vel1 @ nt
                vn2, vt2 = vel2 @ nn, vel2 @ nt

                vn1_ = (vn1 + vn2 + self.e * (vn2 - vn1)) / 2
                vn2_ = (vn1 + vn2 + self.e * (vn1 - vn2)) / 2

                vel1_ = vn1_ * nn + vt1 * nt
                vel2_ = vn2_ * nn + vt2 * nt

                drone1.x[ix.vel] = vel1_
                drone2.x[ix.vel] = vel2_

                overwrap = drone1.R + drone2.R - d2
                shift = overwrap / 2
                shift1 = - shift * nn
                shift2 = shift * nn
                drone1.x[ix.pos] += shift1
                drone2.x[ix.pos] += shift2

    def create_drones(
            self,
            roll: DroneRoll,
            init_state: Union[List, np.ndarray] = None,
            x: float = None, y: float = None,
            u: float = None, v: float = None,
            id_: Union[str, int] = None
    ):
        ix = self.ix

        state = np.zeros(4) if init_state is None else np.asarray(init_state)
        if x is not None:
            state[ix.x] = x
        if y is not None:
            state[ix.y] = y
        if u is not None:
            state[ix.u] = u
        if v is not None:
            state[ix.v] = v

        id_ = str(id_) if id_ is not None else f"drone-{len(self.drones)}"

        drone = Drone2D(roll, dt=self.dt, id_=id_)
        drone.reset(state)
        self.add(drone)
        return drone


    @property
    def states(self):
        return np.array([
            drone.state.copy() for drone in self.drones
        ])

    @property
    def num_drone(self):
        return len(self.drones)
