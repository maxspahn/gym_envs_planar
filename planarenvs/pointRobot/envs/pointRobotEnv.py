import numpy as np
import time
from abc import abstractmethod

from planarenvs.planarCommon.planarEnv import PlanarEnv


class PointRobotEnv(PlanarEnv):

    MAX_VEL = 10
    MAX_POS = 10
    MAX_ACC = 10
    MAX_FOR = 100

    def __init__(self, n=2, dt=0.01, render=False):
        super().__init__(render=render, dt=dt)
        self._n = n
        self._limits = {
            'pos': {'high': np.ones(self._n) * self.MAX_POS, 'low': np.ones(self._n) * -self.MAX_POS},
            'vel': {'high': np.ones(self._n) * self.MAX_VEL, 'low': np.ones(self._n) * -self.MAX_VEL},
            'acc': {'high': np.ones(self._n) * self.MAX_ACC, 'low': np.ones(self._n) * -self.MAX_ACC},
            'for': {'high': np.ones(self._n) * self.MAX_FOR, 'low': np.ones(self._n) * -self.MAX_FOR},
        }
        self._limUpPos = self._limits['pos']['high']
        self._limUpVel = self._limits['vel']['high']
        self._limUpAcc = self._limits['acc']['high']
        self._limUpFor = self._limits['for']['high']
        self.setSpaces()

    def resetLimits(self, **kwargs):
        for key in (kwargs.keys() & self._limits.keys()):
            limitCandidate = kwargs.get(key)
            for limit in (['low', 'high'] & limitCandidate.keys()):
                if limitCandidate[limit].size == self._n:
                    self._limits[key][limit] = limitCandidate[limit]
                else:
                    import logging
                    logging.warning("Ignored reset of limit because the size of the limit is incorrect.")
        self.setSpaces()

    @abstractmethod
    def setSpaces(self):
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
        self.renderCommon(self._limits)
        from gym.envs.classic_control import rendering

        # drawAxis
        self.viewer.draw_line((self._limits['pos']['low'][0], 0), (self._limits['pos']['high'][0], 0))
        self.viewer.draw_line((0, self._limits['pos']['low'][1]), (0, self._limits['pos']['high'][1]))
        # drawPoint
        x = self.state['x'][0:2]
        tf0 = rendering.Transform(rotation=0, translation=(x[0], x[1]))
        joint = self.viewer.draw_circle(.10)
        joint.set_color(.8, .8, 0)
        joint.add_attr(tf0)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
