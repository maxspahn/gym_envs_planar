import numpy as np
from numpy import sin, cos, pi
import time
from abc import abstractmethod

from planarCommon.planarEnv import PlanarEnv


class NLinkReacherEnv(PlanarEnv):

    LINK_LENGTH = 1.0  # [m]
    LINK_MASS = 1.0

    MAX_VEL = 4 * pi
    MAX_POS = pi
    MAX_ACC = 9 * pi
    MAX_TOR = 1000

    def __init__(self, render=False, n=2, dt=0.01):
        super().__init__(render=render, dt=dt)
        self._n = n
        self._limUpPos = np.ones(self._n) * self.MAX_POS
        self._limUpVel = np.ones(self._n) * self.MAX_VEL
        self._limUpAcc = np.ones(self._n) * self.MAX_ACC
        self._limUpTor = np.ones(self._n) * self.MAX_TOR
        self.setSpaces()

    @abstractmethod
    def setSpaces(self):
        pass

    def step(self, a):
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
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def forwardKinematics(self, lastLinkIndex):
        fk = np.array([0.0, 0.2, 0.0])
        for i in range(lastLinkIndex):
            angle = 0.0
            for j in range(i + 1):
                angle += self.state[j]
            fk[0] += np.cos(angle) * self.LINK_LENGTH
            fk[1] += np.sin(angle) * self.LINK_LENGTH
            fk[2] += self.state[i]
        fk[2] += self.state[lastLinkIndex]
        return fk

    def render(self, mode="human"):
        bound = self.LINK_LENGTH * self._n + 0.2
        bounds = [bound, bound]
        self.renderCommon(bounds)

        # axis
        self.viewer.draw_line((-bound, 0), (bound, 0))
        self.renderBase()
        for i in range(self._n):
            self.renderLink(i)
        self.renderEndEffector()
        time.sleep(self.dt())
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def renderBase(self):
        from gym.envs.classic_control import rendering
        base = self.viewer.draw_polygon([(-0.2, 0), (0.0, 0.2), (0.2, 0), (-0.2, 0)])
        baseJoint = self.viewer.draw_circle(0.10)
        baseJoint.set_color(0.8, 0.8, 0)
        tf0 = rendering.Transform(rotation=0, translation=(0.0, 0.2))
        baseJoint.add_attr(tf0)

    def renderLink(self, i):
        from gym.envs.classic_control import rendering
        l, r, t, b = 0, self.LINK_LENGTH, 0.01, -0.01
        fk = self.forwardKinematics(i)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)
        joint = self.viewer.draw_circle(0.10)
        joint.set_color(0.8, 0.8, 0)
        joint.add_attr(tf)

    def renderEndEffector(self):
        from gym.envs.classic_control import rendering
        fk = self.forwardKinematics(self._n)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        eejoint = self.viewer.draw_circle(0.10)
        eejoint.set_color(0.8, 0.8, 0)
        eejoint.add_attr(tf)


