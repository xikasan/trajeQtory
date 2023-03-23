# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
from typing import List, Union

from .base import BaseController
from ..dynamics.manager import DroneManager
from ..dynamics.drone import DroneRoll


class DefenseControllerV0(BaseController):

    def __init__(self, manager: DroneManager, goal_position: Union[List, np.ndarray] = None):
        super().__init__(manager, goal_position=goal_position)

    def __call__(self):
        striker = self.retrieve_drone_by_roll(self.manager.drones, DroneRoll.striker)[0]
        defense = self.retrieve_drone_by_roll(self.manager.drones, DroneRoll.defense)[0]

        intercept_point = (striker.position + self.goal_pos) / 2
        delta_pos = intercept_point - defense.position
        ndelta = delta_pos / nl.norm(delta_pos)
        return ndelta * defense.act_max[0]

