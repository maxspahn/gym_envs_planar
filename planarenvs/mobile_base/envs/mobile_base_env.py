import numpy as np
from abc import abstractmethod

from planarenvs.planar_common.planar_env import PlanarEnv


class MobileBaseEnv(PlanarEnv):

    BASE_HEIGHT = 1.0  # [m]
    BASE_WIDTH = 1.0  # [m]
    LINK_MASS_BASE = 500.0  #: [kg] mass of link 1

    MAX_VEL = 1
    MAX_POS = 5.0
    MAX_ACC = 1.0
    MAX_FOR = 100

    def __init__(self, render=False, dt=0.01):
        super().__init__(render=render, dt=dt)
        self.n = 1
        self._lim_up_pos = np.ones(self._n) * self.MAX_POS
        self._lim_up_vel = np.ones(self._n) * self.MAX_VEL
        self._lim_up_acc = np.ones(self._n) * self.MAX_ACC
        self._lim_up_for = np.ones(self._n) * self.MAX_FOR
        self.set_spaces()

    @abstractmethod
    def set_spaces(self):
        pass

    def _reward(self):
        reward = -1.0 if not self._terminal() else 0.0
        return reward

    def _terminal(self):
        if (
            self._state["joint_state"]["position"][0] > self.MAX_POS
            or self._state["joint_state"]["position"][0] < -self.MAX_POS
        ):
            return True
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render_specific(self, mode="human"):
        bound = self.MAX_POS + 1.0
        self._scale = self.SCREEN_DIM / (2 * bound)
        self._offset = self.SCREEN_DIM / (2 * self._scale)
        self.render_line([-bound - 0.5, 0], [bound + 0.5, 0])

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
