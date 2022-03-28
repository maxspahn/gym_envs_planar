import numpy as np
import time
from abc import abstractmethod
from gym import spaces
import logging

from planarenvs.planar_common.planar_env import PlanarEnv


class PointRobotEnv(PlanarEnv):

    MAX_VEL = 10
    MAX_POS = 10
    MAX_ACC = 10
    MAX_FOR = 100

    def __init__(self, n=2, dt=0.01, render=False):
        super().__init__(render=render, dt=dt)
        self._n = n
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
        self._limUpPos = self._limits["pos"]["high"]
        self._limUpVel = self._limits["vel"]["high"]
        self._limUpAcc = self._limits["acc"]["high"]
        self._limUpFor = self._limits["for"]["high"]
        self.set_spaces()

    def reset_limits(self, **kwargs):
        for key in kwargs.keys() & self._limits.keys():
            limit_candidate = kwargs.get(key)
            for limit in ["low", "high"] & limit_candidate.keys():
                if limit_candidate[limit].size == self._n:
                    self._limits[key][limit] = limit_candidate[limit]
                else:
                    logging.warning("%s",
                        "Incorrect size of limit." +
                        "Resizing ignored."
                    )

        self._limUpPos = self._limits["pos"]["high"]
        self._limUpVel = self._limits["vel"]["high"]
        self._limUpAcc = self._limits["acc"]["high"]
        self._limUpFor = self._limits["for"]["high"]
        self._observation_space.spaces["x"] = spaces.Box(
            low=-self._limUpPos, high=self._limUpPos, dtype=np.float64
        )
        self._observation_space.spaces["xdot"] = spaces.Box(
            low=-self._limUpVel, high=self._limUpVel, dtype=np.float64
        )

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

    def render(self, mode="human"):
        self.render_common(self._limits)
        from gym.envs.classic_control import ( # pylint: disable=import-outside-toplevel
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
        x = self._state["x"][0:2]
        tf0 = rendering.Transform(rotation=0, translation=(x[0], x[1]))
        joint = self._viewer.draw_circle(0.10)
        joint.set_color(0.8, 0.8, 0)
        joint.add_attr(tf0)
        time.sleep(self.dt())

        return self._viewer.render(return_rgb_array=mode == "rgb_array")
