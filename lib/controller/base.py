# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Union

from ..dynamics.manager import DroneManager
from ..dynamics.drone import DroneRoll


class BaseController:

    DEFAULT_GOAL_POS = [9, 5]

    def __init__(self, manager: DroneManager, goal_position: Union[List, np.ndarray] = None, target: Union[int, str] = None):
        self.manager: DroneManager = manager
        if goal_position is None:
            goal_position = self.DEFAULT_GOAL_POS
        goal_position = np.asarray(goal_position)
        self.goal_pos: np.ndarray = goal_position
        self.target = self.find_target_index(target)

    def find_target_index(self, target: Union[int, str]) -> int:
        if target is None:
            return None
        if isinstance(target, int):
            return target
        if isinstance(target, str):
            for i, d in enumerate(self.retrieve_drone_by_roll(self.manager.drones, DroneRoll.defense)):
                if d.id_ == target:
                    return i
        raise ValueError()

    @staticmethod
    def retrieve_drone_by_roll(drones, roll: DroneRoll):
        return [d for d in drones if d.roll is roll]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
