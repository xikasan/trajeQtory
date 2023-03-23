# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
from typing import List, Union

from .base import BaseController
from ..dynamics.manager import DroneManager
from ..dynamics.drone import DroneRoll


class DefenseControllerV0(BaseController):

    def __init__(self, manager: DroneManager, goal_position: Union[List, np.ndarray] = None, target: Union[int, str] = None):
        super().__init__(manager, goal_position=goal_position, target=target)

    def __call__(self):
        striker = self.retrieve_drone_by_roll(self.manager.drones, DroneRoll.striker)[0]
        defenses = self.retrieve_drone_by_roll(self.manager.drones, DroneRoll.defense)

        if self.target is not None:
            defense = defenses[self.target]
            return self.compute_act(striker, defense)

        actions = [self.compute_act(striker, defense) for defense in defenses]
        return actions

    def compute_act(self, striker, defense):
        intercept_point = (striker.position + self.goal_pos) / 2
        delta_pos = intercept_point - defense.position
        ndelta = delta_pos / nl.norm(delta_pos)
        return ndelta * defense.act_max[0]


class DefenseControllerV1(DefenseControllerV0):

    def __init__(self, manager: DroneManager, goal_position: Union[List, np.ndarray] = None, target: Union[int, str] = None):
        super().__init__(manager, goal_position=goal_position, target=target)

    def compute_act(self, striker, defense):
        delta_pos = striker.position - defense.position
        ndelta = delta_pos / nl.norm(delta_pos)
        return ndelta * defense.act_max[0]
