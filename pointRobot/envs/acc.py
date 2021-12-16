import numpy as np
from gym import spaces

from pointRobot.envs.pointRobotEnv import PointRobotEnv


class PointRobotAccEnv(PointRobotEnv):
    def setSpaces(self):
        o_l = np.concatenate((self._limits['pos']['low'], self._limits['vel']['low']))
        o_h = np.concatenate((self._limits['pos']['high'], self._limits['vel']['high']))
        self.observation_space = spaces.Box(low=o_l, high=o_h, dtype=np.float64)
        self.action_space = spaces.Box(
            low=self._limits['acc']['low'], high=self._limits['acc']['high'], dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        vel = x[self._n : self._n * 2]
        xdot = np.concatenate((vel, self.action))
        return xdot
