# -*- coding: utf-8 -*-

import xsim
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from lib.env.single import DroneSoccerSingleEnvV2

alpha = np.pi / 3
cos = np.cos(alpha)
sin = np.sin(alpha)
Ra = np.array([
    [cos, -sin],
    [sin,  cos]
]).T
Rb = np.array([
    [ cos, sin],
    [-sin, cos]
]).T


def rot(angle, vec):
    cos = np.cos(angle)
    sin = np.sin(angle)
    R = np.array([[cos, -sin], [sin, cos]]).T
    return vec @ R


def run():
    env = DroneSoccerSingleEnvV2()
    env.reset()

    # define 2 waypoints, on x=7 and x=8,
    # one point is shortest pass from current position
    waypoint_x = np.array([7, 8])
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

    plt.scatter(*striker_pos)
    plt.scatter(*goal_pos)
    plt.scatter(*waypoints_g.T)
    plt.scatter(*waypoints_a.T)
    plt.scatter(*waypoints_b.T)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()





if __name__ == '__main__':
    run()
