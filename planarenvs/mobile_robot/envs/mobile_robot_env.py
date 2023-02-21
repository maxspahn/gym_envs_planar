import numpy as np
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
            self._state["joint_state"]["position"][0] > self.MAX_POS_BASE
            or self._state["joint_state"]["position"][0] < -self.MAX_POS_BASE
        ):
            return True
        return False

    def _reward(self):
        reward = -1.0 if not self._terminal() else 0.0
        return reward

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render_specific(self, mode="human"):
        bound = self.MAX_POS_BASE + 1.0
        self._scale = self.SCREEN_DIM / (2 * bound)
        self._offset = self.SCREEN_DIM / (2 * self._scale)
        self.render_line([-5.5, 0], [5.5, 0])

        self.render_base()
        for i in range(1, self._n):
            self.render_link(i)
        self.render_end_effector()

    def render_base(self):
        p0 = [self._state["joint_state"]["position"][0], 0.5 * self.BASE_HEIGHT]
        tf_matrix = np.array(((1, 0, p0[0]), (0, 1, p0[1]), (0, 0, 1)))
        l, r, t, b = (
            -0.5 * self.BASE_WIDTH,
            0.5 * self.BASE_WIDTH,
            0.5 * self.BASE_HEIGHT,
            -0.5 * self.BASE_HEIGHT,
        )
        corner_points = [[l, b, 1], [l, t, 1], [r, t, 1], [r, b, 1]]
        transformed_corner_points = []
        for corner_point in corner_points:
            transformed_corner_points.append(
                np.dot(tf_matrix, corner_point)[0:2]
            )
        self.render_polygone(transformed_corner_points)
        self.render_point([p0[0], self.BASE_HEIGHT + 0.2])

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
