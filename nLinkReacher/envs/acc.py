import numpy as np
from gym import spaces

from nLinkReacher.envs.nLinkReacherEnv import NLinkReacherEnv


class NLinkAccReacherEnv(NLinkReacherEnv):
    def setSpaces(self):
        o = np.concatenate((self._limUpPos, self._limUpVel))
        self.observation_space = spaces.Box(low=-o, high=o, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-self._limUpAcc, high=self._limUpAcc, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        u = self.action
        vel = np.array(x[self._n : self._n * 2])
        acc = np.concatenate((vel, u))
        return acc
