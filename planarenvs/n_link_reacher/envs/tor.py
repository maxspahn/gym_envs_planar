import numpy as np
from gym import spaces

from planarenvs.n_link_reacher.envs.n_link_reacher_env import NLinkReacherEnv
from planarenvs.n_link_reacher.resources.createDynamics import createDynamics


class NLinkTorReacherEnv(NLinkReacherEnv):
    def __init__(self, render=False, n=2, dt=0.01, friction=0.0):
        super().__init__(render, n, dt)
        self._dynamics_fun, _, _ = createDynamics(self._n)
        self._friction = friction
        self._g = 10

    def set_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-self._lim_up_pos,
                    high=self._lim_up_pos,
                    dtype=np.float64
                ),
                "xdot": spaces.Box(
                    low=-self._lim_up_vel,
                    high=self._lim_up_vel,
                    dtype=np.float64
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-self._lim_up_tor, high=self._lim_up_tor, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        q = x[0 : self._n]
        qdot = x[self._n : self._n * 2]
        l = [self.LINK_LENGTH for i in range(self._n)]
        m = [self.LINK_MASS for i in range(self._n)]
        tau = self._action
        acc = np.array(
            self._dynamics_fun(q, qdot, l, m, self._g, self._friction, tau)
        )[:, 0]
        return acc
