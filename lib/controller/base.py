# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Union

from ..dynamics.manager import DroneManager
from ..dynamics.drone import DroneRoll


class BaseController:

    DEFAULT_GOAL_POS = [9, 5]

    def __init__(self, manager: DroneManager, goal_position: Union[List, np.ndarray] = None):
        self.manager: DroneManager = manager
        if goal_position is None:
            goal_position = self.DEFAULT_GOAL_POS
        goal_position = np.asarray(goal_position)
        self.goal_pos: np.ndarray = goal_position

    @staticmethod
    def retrieve_drone_by_roll(drones, roll: DroneRoll):
        return [d for d in drones if d.roll is roll]
