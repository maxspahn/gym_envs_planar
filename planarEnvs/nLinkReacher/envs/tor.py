import numpy as np
from gym import spaces

from nLinkReacher.envs.nLinkReacherEnv import NLinkReacherEnv
from nLinkReacher.resources.createDynamics import createDynamics


class NLinkTorReacherEnv(NLinkReacherEnv):
    def __init__(self, render=False, n=2, dt=0.01, friction=0.0):
        super().__init__(render, n, dt)
        self._dynamics_fun, _, _ = createDynamics(self._n)
        self._friction = friction
        self._g = 10

    def setSpaces(self):
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-self._limUpPos, high=self._limUpPos, dtype=np.float64),
            'xdot': spaces.Box(low=-self._limUpVel, high=self._limUpVel, dtype=np.float64),
        })
        self.action_space = spaces.Box(
            low=-self._limUpTor, high=self._limUpTor, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        q = x[0 : self._n]
        qdot = x[self._n : self._n * 2]
        l = [self.LINK_LENGTH for i in range(self._n)]
        m = [self.LINK_MASS for i in range(self._n)]
        tau = self.action
        acc = np.array(self._dynamics_fun(q, qdot, l, m, self._g, self._friction, tau))[
            :, 0
        ]
        return acc
