# -*- coding: utf-8 -*-

import gym
import numpy as np
import time
import numpy.linalg as nl
import cv2

from ..dynamics.drone import Drone2D, IndexDrone2D, DroneRoll
from ..dynamics.manager import DroneManager
from ..controller.base import BaseController
from ..controller.defense import DefenseControllerV0, DefenseControllerV1


class DroneSoccerSingleEnvV0(gym.Env):

    dt = 0.01  # sec | simulation time step
    due = 10.  # sec | max episode time
    MAX_EPISODE_STEP = due // dt
    WORLD_SIZE = np.array([10, 10])

    INIT_POS_RANGE_X = np.array([5, 6])
    INIT_POS_RANGE_Y = np.array([2, 8])
    INIT_DEFENSE_RANGE_X = np.array([7, 8])
    INIT_DEFENSE_RANGE_Y = np.array([3, 7])

    INIT_STRIKER_POS_MIN = np.array([INIT_POS_RANGE_X[0], INIT_POS_RANGE_Y[0]])
    INIT_STRIKER_POS_MAX = np.array([INIT_POS_RANGE_X[1], INIT_POS_RANGE_Y[1]])
    INIT_DEFENSE_POS_MIN = np.array([INIT_DEFENSE_RANGE_X[0], INIT_DEFENSE_RANGE_Y[0]])
    INIT_DEFENSE_POS_MAX = np.array([INIT_DEFENSE_RANGE_X[1], INIT_DEFENSE_RANGE_Y[1]])

    GOAL_R = 1  # m | diameter
    GOAL_THICKNESS = 0.1  # m
    GOAL_POS = np.array([9, 5])

    WINDOW_SIZE = np.array([600, 600])  # [px, px]

    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)

    COLOR_STRIKER = COLOR_GREEN
    COLOR_DEFENSE = COLOR_RED
    COLOR_GOAL = COLOR_BLUE

    def __init__(self):
        self.drone = Drone2D(DroneRoll.striker, dt=self.dt, id_="ball")
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
        return self.drone.state.copy().reshape((1, -1))

    @property
    def states(self):
        return self.state

    @property
    def observation(self):
        return self.state

    @property
    def time(self):
        return np.round(self.step_count * self.dt, 2)


class DroneSoccerSingleEnvV1(gym.Env):

    dt = 0.01  # sec | simulation time step
    due = 5.  # sec | max episode time
    MAX_EPISODE_STEP = due // dt
    WORLD_SIZE = np.array([10, 10])

    INIT_POS_RANGE_X = np.array([2, 3])
    INIT_POS_RANGE_Y = np.array([2, 8])
    INIT_DEFENSE_RANGE_X = np.array([7, 8])
    INIT_DEFENSE_RANGE_Y = np.array([2, 8])

    INIT_STRIKER_POS_MIN = np.array([INIT_POS_RANGE_X[0], INIT_POS_RANGE_Y[0]])
    INIT_STRIKER_POS_MAX = np.array([INIT_POS_RANGE_X[1], INIT_POS_RANGE_Y[1]])
    INIT_DEFENSE_POS_MIN = np.array([INIT_DEFENSE_RANGE_X[0], INIT_DEFENSE_RANGE_Y[0]])
    INIT_DEFENSE_POS_MAX = np.array([INIT_DEFENSE_RANGE_X[1], INIT_DEFENSE_RANGE_Y[1]])

    GOAL_R = 1  # m | diameter
    GOAL_THICKNESS = 0.1  # m
    GOAL_POS = np.array([9, 5])

    WINDOW_SIZE = np.array([600, 600])  # [px, px]

    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)

    COLOR_STRIKER = COLOR_GREEN
    COLOR_DEFENSE = COLOR_RED
    COLOR_GOAL = COLOR_BLUE

    def __init__(self, defense_controller: BaseController = DefenseControllerV0):
        super().__init__()
        self.manager = DroneManager()
        self.defense_controller: BaseController = defense_controller(self.manager, target=0)
        self.step_count = 0

        # rendering
        self.render_factor = None
        self.before_time = None

    def step(self, action):
        self.step_count += 1
        act = np.asarray(action)
        assert act.shape == (2,), f"action shape is expected (2,), but {act.shape} is given."
        action = np.vstack([action, self.defense_controller()])
        self.manager.step(action)

        # over check
        pos = self.manager.drones[0].position
        is_line_over = pos[0] >= self.GOAL_POS[0]
        # episode time check
        is_time_over = self.time > self.due
        # goal check
        pos_goal_y = self.GOAL_POS[1]
        goal_size = self.GOAL_R
        is_between_goal = pos_goal_y - goal_size < pos[self.manager.ix.y] < pos_goal_y + goal_size
        is_goal = is_between_goal and is_line_over

        done = is_goal or is_line_over or is_time_over

        # calc reward
        reward = 0 if not is_goal else 1

        return self.observation, reward, done, {}

    def reset(self, init_state: dict = None):
        self.manager = DroneManager()
        self.render_factor = self.WINDOW_SIZE / self.manager.world_max
        if init_state is None:
            pos = self.generate_random_position(self.INIT_STRIKER_POS_MIN, self.INIT_STRIKER_POS_MAX)
            self.manager.create_drones(DroneRoll.striker, **pos)
            pos = self.generate_random_position(self.INIT_DEFENSE_POS_MIN, self.INIT_DEFENSE_POS_MAX)
            self.manager.create_drones(DroneRoll.defense, **pos)
            return self.states
        for roll, positions in init_state.items():
            for position in positions:
                self.manager.create_drones(roll, x=position[0], y=position[1])
        return self.states

    def render(self, mode="human"):
        img = np.ones((*self.WINDOW_SIZE, 3))
        f = self.f
        # draw drones
        striker = self.manager.drones[0]
        defense = self.manager.drones[1]
        cv2.circle(img, tuple(f(striker.position)), f(x=striker.R), self.COLOR_STRIKER, thickness=5)
        cv2.circle(img, tuple(f(defense.position)), f(x=defense.R), self.COLOR_DEFENSE, thickness=5)
        # draw goal
        corner_shift = np.array([self.GOAL_THICKNESS, self.GOAL_R]) / 2
        cv2.rectangle(img, tuple(f(self.GOAL_POS - corner_shift)), tuple(f(self.GOAL_POS + corner_shift)), self.COLOR_GOAL, thickness=3)
        # draw info
        cv2.putText(img, f"time: {self.time: >5} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
        if mode == "human" or mode == "debug-show":
            cv2.imshow("DroneSoccer", img)
            if self.before_time is None:
                wait_time = self.dt
            else:
                current_time = time.time()
                delta_time = current_time - self.before_time
                wait_time = self.dt - delta_time
            if mode == "debug-show":
                wait_time = -1
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
        return self.manager.states

    @property
    def states(self):
        return self.state

    @property
    def observation(self):
        return self.manager.states.flatten()

    @property
    def time(self):
        return np.round(self.step_count * self.dt, 2)

    @staticmethod
    def generate_random_position(range_min, range_max):
        pos_range = range_max - range_min
        pos = np.random.rand(2) * pos_range + range_min
        return dict(x=pos[0], y=pos[1])

    def f(self, val=None, x=None, y=None):
        if x is not None:
            return int(x * self.render_factor[0])
        if y is not None:
            return int(y * self.render_factor[1])
        if val is not None:
            assert val.shape == (2,), f"val.shape is expected (2,), but {val.shape} is given."
            val = np.asarray(val)
            val = val * self.render_factor
            return val.astype(int)
        raise ValueError("Not appropriate data is given")


class DroneSoccerSingleEnvV2(DroneSoccerSingleEnvV1):

    def __init__(
            self,
            defense_controller1: BaseController = DefenseControllerV0,
            defense_controller2: BaseController = DefenseControllerV1
    ):
        super().__init__()
        self.manager = DroneManager()
        self.defense_controller1_class = defense_controller1
        self.defense_controller2_class = defense_controller2
        self.defense_controller1: BaseController = defense_controller1(self.manager, target=0)
        self.defense_controller2: BaseController = defense_controller2(self.manager, target=1)
        self.step_count = 0

        # rendering
        self.render_factor = None
        self.before_time = None

    def step(self, action):
        self.step_count += 1
        act = np.asarray(action)
        assert act.shape == (2,), f"action shape is expected (2,), but {act.shape} is given."
        action = np.vstack([action, self.defense_controller1(), self.defense_controller2()])
        # action = np.vstack([action, np.zeros_like(action), np.zeros_like(action)])
        self.manager.step(action)

        # over check
        pos = self.manager.drones[0].position
        is_line_over = pos[0] >= self.GOAL_POS[0]
        # episode time check
        is_time_over = self.time > self.due
        # goal check
        pos_goal_y = self.GOAL_POS[1]
        goal_size = self.GOAL_R
        is_between_goal = pos_goal_y - goal_size < pos[self.manager.ix.y] < pos_goal_y + goal_size
        is_goal = is_between_goal and is_line_over

        done = is_goal or is_line_over or is_time_over

        # calc reward
        reward = 0 if not is_goal else 1

        return self.observation, reward, done, {}

    def reset(self, init_state: dict = None):
        self.manager = DroneManager()
        self.render_factor = self.WINDOW_SIZE / self.manager.world_max

        self.defense_controller1 = self.defense_controller1_class(self.manager, target=1)
        self.defense_controller2 = self.defense_controller2_class(self.manager, target=1)

        # initial condition is not given = Random start
        if init_state is None:
            pos = self.generate_random_position(self.INIT_STRIKER_POS_MIN, self.INIT_STRIKER_POS_MAX)
            self.manager.create_drones(DroneRoll.striker, **pos)
            pos = self.generate_random_position(self.INIT_DEFENSE_POS_MIN, self.INIT_DEFENSE_POS_MAX)
            self.manager.create_drones(DroneRoll.defense, **pos)
            pos = self.generate_random_position(self.INIT_DEFENSE_POS_MIN, self.INIT_DEFENSE_POS_MAX)
            self.manager.create_drones(DroneRoll.defense, **pos)
            return self.observation

        # initial condition is given
        for roll, positions in init_state.items():
            for position in positions:
                self.manager.create_drones(roll, x=position[0], y=position[1])
        return self.observation

    def render(self, mode="human"):
        img = np.ones((*self.WINDOW_SIZE, 3))
        f = self.f
        # draw drones
        striker = BaseController.retrieve_drone_by_roll(self.manager.drones, DroneRoll.striker)[0]
        defenses = BaseController.retrieve_drone_by_roll(self.manager.drones, DroneRoll.defense)
        cv2.circle(img, tuple(f(striker.position)), f(x=striker.R), self.COLOR_STRIKER, thickness=5)
        [
            cv2.circle(img, tuple(f(d.position)), f(x=d.R), self.COLOR_DEFENSE, thickness=5)
            for d in defenses
        ]
        # draw goal
        corner_shift = np.array([self.GOAL_THICKNESS, self.GOAL_R]) / 2
        cv2.rectangle(img, tuple(f(self.GOAL_POS - corner_shift)), tuple(f(self.GOAL_POS + corner_shift)), self.COLOR_GOAL, thickness=3)
        # draw info
        cv2.putText(img, f"time: {self.time: >5} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
        if mode == "human" or mode == "debug":
            cv2.imshow("DroneSoccer", img)
            if self.before_time is None:
                wait_time = self.dt
            else:
                current_time = time.time()
                delta_time = current_time - self.before_time
                wait_time = self.dt - delta_time
            if mode == "debug":
                wait_time = -1
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
        return self.manager.states

    @property
    def states(self):
        return self.state

    @property
    def observation(self):
        return self.manager.states.flatten()

    @property
    def time(self):
        return np.round(self.step_count * self.dt, 2)

    @staticmethod
    def generate_random_position(range_min, range_max):
        pos_range = range_max - range_min
        pos = np.random.rand(2) * pos_range + range_min
        return dict(x=pos[0], y=pos[1])

    def f(self, val=None, x=None, y=None):
        if x is not None:
            return int(x * self.render_factor[0])
        if y is not None:
            return int(y * self.render_factor[1])
        if val is not None:
            assert val.shape == (2,), f"val.shape is expected (2,), but {val.shape} is given."
            val = np.asarray(val)
            val = val * self.render_factor
            return val.astype(int)
        raise ValueError("Not appropriate data is given")
