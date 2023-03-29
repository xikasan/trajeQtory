# -*- coding: utf-8 -*-

import cv2
import xsim
import numpy as np
import numpy.linalg as nl
import pandas as pd
import matplotlib.pyplot as plt

from lib.env.single import DroneSoccerSingleEnvV2
from lib.dynamics.drone import DroneRoll
from lib.controller.waypoint import make_waypoints_of_3_path
from lib.controller.base import BaseController


INITIAL_POSITIONS = {
    DroneRoll.striker: [[3, 5]],
    DroneRoll.defense: [[8, 4], [8, 6]]
}


class Launcher:

    REG = 0.1

    def __init__(self, env):
        self.env = env
        self._init_positions = None
        self._waypoints = None

    def reset(self, initial_positions=None):
        if initial_positions is not None:
            self.set_initial_positions(initial_positions)
        self.env.reset(self._init_positions)

    def run(self):
        striker = BaseController.retrieve_drone_by_roll(self.env.manager.drones, DroneRoll.striker)[0]
        log = xsim.Logger()
        done = False
        while not done:
            if self.is_reach_next_waypoint(striker) or self.is_over_run_next_waypoint(striker):
                self._waypoints = self._waypoints[1:]
                print("- "*60)
                print("check point")
            normal_waypoint = self.delta_waypoint(striker) / self.distance_waypoint(striker)
            # action = normal_waypoint * 5
            action = self.delta_waypoint(striker)
            # print("act:", action)

            log.store(time=self.env.time, x=striker.position[0], y=striker.position[1]).flush()

            _, reward, done, _ = self.env.step(action)

            self.env.render()
        ret = xsim.Retriever(log)
        res = pd.DataFrame(dict(
            time=ret.time(),
            x=ret.x(),
            y=ret.y()
        ))
        return res

    def is_reach_next_waypoint(self, striker):
        distance_waypoint = self.distance_waypoint(striker)
        return distance_waypoint < self.REG

    def is_over_run_next_waypoint(self, striker):
        waypoint_x = self._waypoints[0][0]
        striker_x = striker.position[0]
        return striker_x > waypoint_x + 0.2

    def delta_waypoint(self, striker):
        return self._waypoints[0] - striker.position

    def distance_waypoint(self, striker):
        return nl.norm(self.delta_waypoint(striker))

    def set_initial_positions(self, vals):
        self._init_positions = vals

    @property
    def initial_positions(self):
        return self._init_positions

    def set_waypoints(self, vals):
        vals.append(env.GOAL_POS)
        self._waypoints = vals

    @property
    def waypoints(self):
        return self._waypoints


# prepare env and launcher
env = DroneSoccerSingleEnvV2()
lch = Launcher(env)
# lch.reset(INITIAL_POSITIONS)
lch.reset()

# make waypoint list for situation
waypoint_set = make_waypoints_of_3_path(env)
# select waypoint
select_indices = np.random.randint(0, len(waypoint_set), size=2)
# select_indices = [2, 2]
waypoints = [waypoint_set[select_index][i] for i, select_index in enumerate(select_indices)]
lch.set_waypoints(waypoints)

# - - - - - - - - - - - - - - - - - - - - - - - -
result = lch.run()

# - - - - - - - - - - - - - - - - - - - - - - - -
# draw field
drones = env.manager.drones

fig, ax = plt.subplots()
for drone in BaseController.retrieve_drone_by_roll(drones, DroneRoll.striker):
    ax.scatter(*drone.position, label="striker")
for drone in BaseController.retrieve_drone_by_roll(drones, DroneRoll.defense):
    ax.scatter(*drone.position, label="defense")
ax.scatter(*np.vstack(waypoints).T, label="waypoint")

ax.plot(result["x"].to_numpy(), result["y"].to_numpy())

npR = np.array([0, env.GOAL_R])
goal_line = np.vstack([env.GOAL_POS - npR, env.GOAL_POS + npR])
ax.plot(*goal_line.T)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.legend()
plt.show()
