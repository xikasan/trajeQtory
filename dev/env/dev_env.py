# -*- coding: utf-8 -*-

import xsim
import cv2
import numpy as np
import numpy.linalg as nl
import pandas as pd
import matplotlib.pyplot as plt

from lib.env.single import DroneSoccerSingleEnvV0, DroneSoccerSingleEnvV1


def run_v0():
    env = DroneSoccerSingleEnvV0()
    env.reset()

    state = env.state
    pos_drone = state[:2]
    pos_goal = env.goal_pos
    error = pos_goal - pos_drone
    action = error / np.linalg.norm(error) * 10

    done = False
    while not done:
        obs, reward, done, _ = env.step(action)
        env.render()


def run_v1():
    env = DroneSoccerSingleEnvV1()
    env.reset()

    done = False
    while not done:
        states = env.states
        striker_pos = states[0, :2]
        error_to_goal = env.GOAL_POS - striker_pos
        action = error_to_goal / nl.norm(error_to_goal) * 10
        obs, reward, done, _ = env.step(action)
        env.render()
    print(reward)


if __name__ == '__main__':
    run_v1()
