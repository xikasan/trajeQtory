# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from lib.controller.waypoint import make_waypoints_of_3_path
from lib.env.single import DroneSoccerSingleEnvV2
from lib.env.launch import Launcher

env = DroneSoccerSingleEnvV2()
lan = Launcher(env)
lan.reset()

# waypoint definition
wp_sets = make_waypoints_of_3_path(env)
wp_indices = np.random.randint(0, len(wp_sets), size=2)
wps = [wp_sets[wp_index][i] for i, wp_index in enumerate(wp_indices)]
lan.set_waypoints(wps)

# run
result = lan.run()

fig, ax = lan.plot(result)
plt.show()
