import numpy as np
import time
from abc import abstractmethod
from gym import spaces
import logging

from planarenvs.planar_common.planar_env import PlanarEnv
from planarenvs.point_robot.envs.point_robot_env import PointRobotEnv


class MultiPointRobotsEnv(PlanarEnv):

    MAX_VEL = 10
    MAX_POS = 10
    MAX_ACC = 10
    MAX_FOR = 100

    def __init__(self, number_agents=4, dt=0.01, render=False):
        super().__init__(render=render, dt=dt)
        self._n = number_agents * 2
        self._number_robots = int(self._n/2)
        self._limits = {
            "pos": {
                "high": np.ones(self._n) * self.MAX_POS,
                "low": np.ones(self._n) * -self.MAX_POS,
            },
            "vel": {
                "high": np.ones(self._n) * self.MAX_VEL,
                "low": np.ones(self._n) * -self.MAX_VEL,
            },
            "acc": {
                "high": np.ones(self._n) * self.MAX_ACC,
                "low": np.ones(self._n) * -self.MAX_ACC,
            },
            "for": {
                "high": np.ones(self._n) * self.MAX_FOR,
                "low": np.ones(self._n) * -self.MAX_FOR,
            },
        }
        self._lim_up_pos = self._limits["pos"]["high"]
        self._lim_up_vel = self._limits["vel"]["high"]
        self._lim_up_acc = self._limits["acc"]["high"]
        self._lim_up_for = self._limits["for"]["high"]
        self.set_spaces()

    @abstractmethod
    def set_spaces(self):
        pass

    def _reward(self):
        reward = -1.0 if not self._terminal() else 0.0
        return reward

    def _terminal(self):
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render_single_robot(self, index, mode="human"):
        self.render_common(self._limits)
        from gym.envs.classic_control import (  # pylint: disable=import-outside-toplevel
            rendering,
        )

        # drawAxis
        self._viewer.draw_line(
            (self._limits["pos"]["low"][0], 0),
            (self._limits["pos"]["high"][0], 0),
        )
        self._viewer.draw_line(
            (0, self._limits["pos"]["low"][1]),
            (0, self._limits["pos"]["high"][1]),
        )
        # drawPoint
        x = self._state["x"][index*2:index*2+2]
        tf0 = rendering.Transform(rotation=0, translation=(x[0], x[1]))
        joint = self._viewer.draw_circle(0.10)
        joint.set_color(0.8, 0.8, 0)
        joint.add_attr(tf0)
        time.sleep(self.dt())

    def render(self, mode="human"):
        for i in range(self._number_robots):
            self.render_single_robot(i, mode=mode)
        return self._viewer.render(return_rgb_array=mode == "rgb_array")
