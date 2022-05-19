from matplotlib.cbook import ls_mapper
import numpy as np
from numpy import float32, pi
import time
from abc import abstractmethod
from scipy.spatial import distance

from planarenvs.planar_common.planar_env import PlanarEnv
from forwardkinematics.planarFks.planarArmFk import PlanarArmFk


class NLinkReacherEnv(PlanarEnv):

    LINK_LENGTH = 1.0  # [m]
    LINK_MASS = 1.0

    MAX_VEL = 4 * pi
    MAX_POS = pi
    MAX_ACC = 9 * pi
    MAX_TOR = 0.1

    def __init__(self, render=False, n=2, dt=0.01):
        super().__init__(render=render, dt=dt)
        self.n = n
        self._limUpPos = np.ones(self._n, dtype=float32) * self.MAX_POS
        self._limUpVel = np.ones(self._n, dtype=float32) * self.MAX_VEL
        self._limUpAcc = np.ones(self._n, dtype=float32) * self.MAX_ACC
        self._limUpTor = np.ones(self._n, dtype=float32) * self.MAX_TOR
        self.set_spaces()
        self._fk = PlanarArmFk(self._n)

    @abstractmethod
    def set_spaces(self):
        pass

    def _terminal(self):
        current_position = tuple(self._fk.numpy(
            self._state["x"], len(self._state["x"]), True))
        goal_position = tuple(self._goals[0]._contentDict["desired_position"])
        epsilon = self._goals[0]._contentDict["epsilon"]
        gap = distance.euclidean(current_position, goal_position)
        if self._emergency_stop:
            return True
        return gap <= epsilon

    def _reward(self):
        current_position = tuple(self._fk.numpy(
            self._state["x"], len(self._state["x"]), True))
        goal_position = tuple(self._goals[0]._contentDict["desired_position"])
        epsilon = self._goals[0]._contentDict["epsilon"]
        if self._emergency_stop:
            return -10
        gap = distance.euclidean(current_position, goal_position)
        reward = 1 if gap <= epsilon else 0.0
        return reward

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human"):
        bound = self.LINK_LENGTH * self._n + 0.2
        bounds = [bound, bound]
        self.render_common(bounds)

        # axis
        self._viewer.draw_line((-bound, 0), (bound, 0))
        self.render_base()
        for i in range(self._n):
            self.render_link(i)
        self.render_end_effector()
        time.sleep(self.dt())
        return self._viewer.render(return_rgb_array=mode == "rgb_array")

    def render_base(self):
        from gym.envs.classic_control import rendering  # pylint: disable=import-outside-toplevel

        base = self._viewer.draw_polygon(
            [(-0.2, 0), (0.0, 0.2), (0.2, 0), (-0.2, 0)]
        )
        tf0 = rendering.Transform(rotation=0, translation=(0.0, -0.2))
        base.add_attr(tf0)

    def render_link(self, i):
        from gym.envs.classic_control import rendering  # pylint: disable=import-outside-toplevel

        l, r, t, b = 0, self.LINK_LENGTH, 0.01, -0.01
        fk = self._fk.fk(self._state["x"], i)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        link = self._viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)
        joint = self._viewer.draw_circle(0.10)
        joint.set_color(0.8, 0.8, 0)
        joint.add_attr(tf)

    def render_end_effector(self):
        from gym.envs.classic_control import rendering  # pylint: disable=import-outside-toplevel

        fk = self._fk.fk(self._state["x"], self._n)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        eejoint = self._viewer.draw_circle(0.10)
        eejoint.set_color(0.8, 0.8, 0)
        eejoint.add_attr(tf)
