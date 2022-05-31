import numpy as np
import time
from abc import abstractmethod

from forwardkinematics.planarFks.mobileRobotFk import MobileRobotFk

from planarenvs.planar_common.planar_env import PlanarEnv


class MobileRobotEnv(PlanarEnv):

    BASE_HEIGHT = 1.0  # [m]
    BASE_WIDTH = 1.0  # [m]
    LINK_LENGTH = 1.0  # [m]

    MAX_VEL_BASE = 1
    MAX_POS_BASE = 5.0
    MAX_ACC_BASE = 10.0
    MAX_FOR_BASE = 100.0
    MAX_VEL = 4 * np.pi
    MAX_POS = np.pi
    MAX_ACC = 8 * np.pi
    MAX_TOR = 1000

    def __init__(self, render=False, n=2, dt=0.01):
        super().__init__(render=render, dt=dt)
        self.n = n
        self._lim_up_pos = self.join_limits(
            np.array([self.MAX_POS_BASE]), np.ones(self._n - 1) * self.MAX_POS
        )
        self._lim_up_vel = self.join_limits(
            np.array([self.MAX_VEL_BASE]), np.ones(self._n - 1) * self.MAX_VEL
        )
        self._lim_up_acc = self.join_limits(
            np.array([self.MAX_ACC_BASE]), np.ones(self._n - 1) * self.MAX_ACC
        )
        self._lim_up_tor = self.join_limits(
            np.array([self.MAX_FOR_BASE]), np.ones(self._n - 1) * self.MAX_TOR
        )
        self.set_spaces()
        self._fk = MobileRobotFk(self._n, baseHeight=self.BASE_HEIGHT)

    def join_limits(self, lim_bas, lim_arm):
        return np.concatenate((lim_bas, lim_arm))

    @abstractmethod
    def set_spaces(self):
        pass

    def _get_ob(self):
        return self._state

    def _terminal(self):
        if (
            self._state["x"][0] > self.MAX_POS_BASE
            or self._state["x"][0] < -self.MAX_POS_BASE
        ):
            return True
        return False

    def _reward(self):
        reward = -1.0 if not self._terminal() else 0.0
        return reward

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human"):
        bound = self.MAX_POS_BASE + 1.0
        bounds = [bound, bound]
        self.render_common(bounds)

        # drawAxis
        self._viewer.draw_line((-5.5, 0), (5.5, 0))

        self.render_base()
        for i in range(1, self._n):
            self.render_link(i)
        self.render_end_effector()
        time.sleep(self.dt())
        return self._viewer.render(return_rgb_array=mode == "rgb_array")

    def render_base(self):
        from gym.envs.classic_control import rendering #pylint: disable=import-outside-toplevel

        l, r, t, b = (
            -0.5 * self.BASE_WIDTH,
            0.5 * self.BASE_WIDTH,
            0.5 * self.BASE_HEIGHT,
            -0.5 * self.BASE_HEIGHT,
        )
        tf = rendering.Transform(
            rotation=0,
            translation=(self._state["x"][0], 0.5 * self.BASE_HEIGHT),
        )
        link = self._viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)
        base = self._viewer.draw_polygon(
            [(-0.2, -0.2), (0.0, 0.0), (0.0, 0.0), (0.2, -0.2)]
        )
        base_joint = self._viewer.draw_circle(0.10)
        base_joint.set_color(0.8, 0.8, 0)
        tf0 = rendering.Transform(
            rotation=0,
            translation=(self._state["x"][0], self.BASE_HEIGHT + 0.2),
        )
        base_joint.add_attr(tf0)
        base.add_attr(tf0)

    def render_link(self, i):
        from gym.envs.classic_control import rendering #pylint: disable=import-outside-toplevel

        l, r, t, b = 0, self.LINK_LENGTH, 0.01, -0.01
        fk = self._fk.fk(self._state["x"], i)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        link = self._viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.8, 0.8)
        joint = self._viewer.draw_circle(0.10)
        joint.set_color(0.8, 0.8, 0)
        link.add_attr(tf)
        joint.add_attr(tf)

    def render_end_effector(self):
        from gym.envs.classic_control import rendering #pylint: disable=import-outside-toplevel

        fk = self._fk.fk(self._state["x"], self._n)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        eejoint = self._viewer.draw_circle(0.10)
        eejoint.set_color(0.8, 0.8, 0)
        eejoint.add_attr(tf)
