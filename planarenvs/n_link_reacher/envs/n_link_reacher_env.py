import numpy as np
from numpy import pi
from abc import abstractmethod

from planarenvs.planar_common.planar_env import PlanarEnv
from forwardkinematics.planarFks.planarArmFk import PlanarArmFk


class NLinkReacherEnv(PlanarEnv):

    LINK_LENGTH = 1.0  # [m]
    LINK_MASS = 1.0

    MAX_VEL = 4 * pi
    MAX_POS = pi
    MAX_ACC = 9 * pi
    MAX_TOR = 1000

    def __init__(self, render=False, n=2, dt=0.01):
        super().__init__(render=render, dt=dt)
        self.n = n
        self._lim_up_pos = np.ones(self._n) * self.MAX_POS
        self._lim_up_vel = np.ones(self._n) * self.MAX_VEL
        self._lim_up_acc = np.ones(self._n) * self.MAX_ACC
        self._lim_up_tor = np.ones(self._n) * self.MAX_TOR
        self.set_spaces()
        self._fk = PlanarArmFk(self._n)

    @abstractmethod
    def set_spaces(self):
        pass

    def _terminal(self):
        return False

    def _reward(self):
        reward = -1.0 if not self._terminal() else 0.0
        return reward

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render_specific(self, mode="human"):
        bound = self.LINK_LENGTH * self._n + 0.2
        self._scale = self.SCREEN_DIM / (2 * bound)
        self._offset = self.SCREEN_DIM / (2 * self._scale)
        self.render_line([-bound, 0], [bound, 0])
        self.render_base()
        for i in range(self._n):
            self.render_link(i)
        self.render_end_effector()

    def render_base(self):
        self.render_point([0.0, 0.0], color=(0, 0, 0))

    def render_link(self, i):
        fk = self._fk.fk(
            self._state["joint_state"]["position"], i, positionOnly=False
        )
        c, s = np.cos(fk[2]), np.sin(fk[2])
        tf_matrix = np.array(((c, -s, fk[0]), (s, c, fk[1]), (0, 0, 1)))
        l, r, t, b = 0, self.LINK_LENGTH, 0.01, -0.01
        corner_points = [[l, b, 1], [l, t, 1], [r, t, 1], [r, b, 1]]
        transformed_corner_points = []
        for corner_point in corner_points:
            transformed_corner_points.append(
                np.dot(tf_matrix, corner_point)[0:2]
            )
        self.render_polygone(transformed_corner_points, color=(0, 0, 0))
        self.render_point(fk[0:2])

    def render_end_effector(self):
        fk = self._fk.fk(self._state["joint_state"]["position"], self._n)
        self.render_point(fk[0:2], color=(0.8 * 255, 0.8 * 255, 0.0 * 255))
