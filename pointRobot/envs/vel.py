import numpy as np
from gym import spaces

from pointRobot.envs.pointRobotEnv import PointRobotEnv


class PointRobotVelEnv(PointRobotEnv):
    def setSpaces(self):
        o_l = np.concatenate((self._limits['pos']['low'], self._limits['vel']['low']))
        o_h = np.concatenate((self._limits['pos']['high'], self._limits['vel']['high']))
        self.observation_space = spaces.Box(low=o_l, high=o_h, dtype=np.float64)
        self.action_space = spaces.Box(
            low=self._limits['vel']['low'], high=self._limits['vel']['high'], dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        vel = self.action
        acc = np.zeros(self._n)
        self.state[self._n: 2 * self._n] = vel
        xdot = np.concatenate((vel, acc))
        return xdot
