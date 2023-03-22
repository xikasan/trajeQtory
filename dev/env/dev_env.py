# -*- coding: utf-8 -*-

import xsim
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.env.single import DroneSoccerSingleEnv


def run():
    env = DroneSoccerSingleEnv()
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


if __name__ == '__main__':
    run()
