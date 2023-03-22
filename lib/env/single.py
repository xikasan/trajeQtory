# -*- coding: utf-8 -*-

import gym
import numpy as np
import time
import numpy.linalg as nl
import cv2

from ..dynamics.drone import Drone2D, IndexDrone2D


class DroneSoccerSingleEnv(gym.Env):

    dt = 0.01  # sec | simulation time step
    due = 30  # sec | max episode time
    goal_pos = np.array([9, 5])  # m
    goal_size = 1  # m | in diameter
    window_size = 600  # px | window height and width
    max_step_count = due // dt

    def __init__(self):
        self.drone = Drone2D(dt=self.dt, id_="ball")
        self.step_count = 0
        self.before_time = None

    def step(self, action):
        self.step_count += 1
        act = np.asarray(action)
        assert act.shape == (2,), f"action shape is expected (2,), but {act.shape} is given."
        self.drone(act)

        # over check
        pos = self.drone.position
        is_line_over = pos[0] >= self.goal_pos[0]
        # episode time check
        is_time_over = self.time > self.due
        # goal check
        pos_goal_y = self.goal_pos[1]
        goal_size = self.goal_size
        is_between_goal = pos_goal_y - goal_size < pos[self.drone.ix.y] < pos_goal_y + goal_size
        is_goal = is_between_goal and is_line_over

        done = is_goal or is_line_over or is_time_over

        # calc reward
        reward = 0 if not is_goal else 1

        return self.state, reward, done, {}

    def reset(self):
        ix = self.drone.ix
        state = np.zeros(4)
        state[ix.x] = np.random.rand() * 3 + 5
        state[ix.y] = np.random.rand() * 6 + 2
        self.drone.reset(xinit=state)
        return self.state

    def render(self, mode="human"):
        f = self.window_size / self.drone.pos_max[0]
        img = np.ones((self.window_size, self.window_size, 3))
        cv2.circle(img, tuple((self.drone.position * f).astype(int)), int(self.drone.R * f), (255, 0, 0), thickness=5)
        cv2.circle(img, tuple((self.goal_pos * f).astype(int)), int(self.goal_size / 2 * f), (0, 0, 255), thickness=1)
        cv2.putText(img, f"time: {self.time: >5} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
        if mode == "human":
            cv2.imshow("DroneSoccer", img)
            if self.before_time is None:
                wait_time = self.dt
            else:
                current_time = time.time()
                delta_time = current_time - self.before_time
                wait_time = self.dt - delta_time
            cv2.waitKey(int(wait_time * 1000))
            self.before_time = time.time()
            return
        return img

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        return seed

    @property
    def state(self):
        return self.drone.state.copy()

    @property
    def observation(self):
        return self.state

    @property
    def time(self):
        return np.round(self.step_count * self.dt, 2)
