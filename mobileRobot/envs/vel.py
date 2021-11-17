import numpy as np
from gym import spaces

from mobileRobot.envs.mobileRobotEnv import MobileRobotEnv

class MobileRobotVelEnv(MobileRobotEnv):

    def setSpaces(self):
        o = np.concatenate((self._limUpPos, self._limUpVel))
        self.observation_space = spaces.Box(low=-o, high=o, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-self._limUpAcc, high=self._limUpAcc, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        self.state[self._n: 2 * self._n] = self.action
        vel = self.action
        acc = np.zeros(self._n)
        xdot = np.concatenate((vel, acc))
        return xdot
