# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from lib.dynamics.manager import DroneManager
from lib.dynamics.drone import DroneRoll
from lib.controller.defense import DefenseControllerV0


def run():
    manager = DroneManager()
    manager.create_drones(DroneRoll.striker, x=0, y=0)
    manager.create_drones(DroneRoll.defense, x=1, y=0)

    controller = DefenseControllerV0(manager)
    controller()


if __name__ == '__main__':
    run()
