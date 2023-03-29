# -*- coding: utf-8 -*-

import numpy as np

from lib.dynamics.manager import DroneRoll
from lib.env.single import DroneSoccerSingleEnvV1, DroneSoccerSingleEnvV2
from lib.controller.waypoint import make_waypoints_of_3_path, WaypointController


def run_solo():
    env = DroneSoccerSingleEnvV1()
    env.reset()

    waypoints = make_waypoints_of_3_path(env)
    waypoints = waypoints[1]

    controller = WaypointController(env.manager, env.GOAL_POS)
    controller.set_target(DroneRoll.striker, id_=0)
    controller.waypoints(waypoints)

    done = False
    while not done:
        states = env.states
        action = controller()

        obs, reward, done, _ = env.step(action)
        env.render()
    print(reward)



if __name__ == '__main__':
    run_solo()
