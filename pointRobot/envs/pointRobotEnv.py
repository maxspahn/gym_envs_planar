import numpy as np
import time
from abc import abstractmethod

from planarCommon.planarEnv import PlanarEnv

class PointRobotEnv(PlanarEnv):

    MAX_VEL = 10
    MAX_POS = 10
    MAX_ACC = 10
    MAX_FOR = 100

    def __init__(self, n=2, dt=0.01, render=False):
        self._n = n
        self.viewer = None
        self._limUpPos = np.ones(self._n) * self.MAX_POS
        self._limUpVel = np.ones(self._n) * self.MAX_VEL
        self._limUpAcc = np.ones(self._n) * self.MAX_ACC
        self._limUpFor = np.ones(self._n) * self.MAX_FOR
        self.setSpaces()
        self.state = None
        self.seed()
        self._dt = dt
        self._render = render

    @abstractmethod
    def setSpaces(self):
        pass

    def reset(self, pos=None, vel=None):
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        _ = self.continuous_dynamics(self.state, 0)
        ns = self.integrate()
        self.state = ns
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        return self.state

    def _terminal(self):
        s = self.state
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state
        if s is None:
            return None

        bound = 5.0
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        self.viewer.draw_line((-bound, 0), (bound, 0))
        self.viewer.draw_line((0, -bound), (0, bound))
        x = s[0]
        y = 0.0
        if self._n == 2:
            y = s[1]
        tf0 = rendering.Transform(rotation=0, translation=(x, y))
        joint = self.viewer.draw_circle(.10)
        joint.set_color(.8, .8, 0)
        joint.add_attr(tf0)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
