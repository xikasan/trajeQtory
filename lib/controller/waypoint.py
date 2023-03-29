# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
from typing import List, Union

from .base import BaseController
from ..dynamics.manager import DroneManager
from ..dynamics.drone import DroneRoll


def make_waypoints_of_3_path(env):
    # define 2 waypoints, on x=5 and x=7,
    # one point is shortest pass from current position
    waypoint_x = np.array([5, 7])
    states = env.states
    striker_pos = states[0, :2]

    # targets
    goal_pos = env.GOAL_POS
    corner_a_pos = env.WORLD_SIZE
    corner_b_pos = np.array([env.WORLD_SIZE[0], 0])

    # step x
    waypoint_deltaxs = waypoint_x - striker_pos[0]
    waypoint_deltaxs = waypoint_deltaxs[np.newaxis, :].T

    def make_waypoints(origin, target):
        delta_target = target - origin
        ntarget = delta_target / delta_target[0]
        return waypoint_deltaxs * ntarget + striker_pos

    # direct
    waypoints_g = make_waypoints(striker_pos, goal_pos)
    # corners
    waypoints_a = make_waypoints(striker_pos, corner_a_pos)
    waypoints_b = make_waypoints(striker_pos, corner_b_pos)
    return [waypoints_g, waypoints_a, waypoints_b]


class WaypointController(BaseController):

    REG = 0.2  # m | clearance distance for waypoint

    def __init__(
            self,
            manager: DroneManager,
            goal_position: Union[List, np.ndarray] = None,
            target: Union[int, str] = None,
            roll: DroneRoll = None,
            gain: Union[float, List[float], np.ndarray] = 1.
    ):
        super().__init__(manager, goal_position=goal_position, target=target)
        self.target_roll: DroneRoll = roll
        self._waypoints = None
        self.K = gain if hasattr(gain, "__len__") else np.asarray(gain)

    def __call__(self):
        pos = self.target.position
        print(len(self._waypoints), self._waypoints[0], pos)
        # for waypoint
        if len(self._waypoints) > 0:
            wp_point = self._waypoints[0, :]
            action = self.compute_act(pos, wp_point)
            delta_wp = wp_point - pos
            if nl.norm(delta_wp) <= self.REG:
                self._waypoints = self._waypoints[1:]
            return action
        # for goal
        return self.compute_act(pos, self.goal_pos)

    def set_target(self, roll: DroneRoll, id_: Union[int, str] = None):
        drones = self.retrieve_drone_by_roll(
            self.manager.drones, roll=roll
        )
        id_ = 0 if id_ is None else self.find_target_index(id_)
        self.target = drones[id_]

    def waypoints(self, waypoints):
        waypoints = np.asarray(waypoints)
        self._waypoints = waypoints

    def compute_act(self, drone_pos, waypoint_pos):
        delta_pos = waypoint_pos - drone_pos
        ndelta = delta_pos / nl.norm(delta_pos)
        return ndelta * self.K
