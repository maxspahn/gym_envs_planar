import numpy as np
from gym import spaces

from mobileBase.envs.mobileBaseEnv import MobileBaseEnv


class MobileBaseVelEnv(MobileBaseEnv):

    def setSpaces(self):
        o = np.concatenate((self._limUpPos, self._limUpVel))
        self.observation_space = spaces.Box(low=-o, high=o, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-self._limUpVel, high=self._limUpVel, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        vel = self.action
        acc = np.zeros(self._n)
        self.state[self._n: 2 * self._n] = vel
        return np.concatenate((vel, acc))

