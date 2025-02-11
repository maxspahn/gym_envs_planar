import numpy as np
from abc import abstractmethod

#from forwardkinematics.planarFks.mobileRobotFk import MobileRobotFk
from forwardkinematics.planarFks.planarArmFk import PlanarArmFk

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
    _limits = {
        "pos": {
            "high": np.array([5, 5]),
            "low": np.array([-5, -5]),
        }
    }
    _joint_color = (0.1 * 255, 0.2 * 255, 0.9 * 255)

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
        self._fk = PlanarArmFk(self._n-1)

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
        self.render_polygone(transformed_corner_points, color=self._joint_color)
        self.render_point([p0[0], self.BASE_HEIGHT + 0.2], color=self._joint_color)

    def compute_fk(self, i: int) -> np.ndarray:
        q_current = self._state["joint_state"]["position"]
        q_base = q_current[0]
        fk_base = np.array([[1, 0, q_base], [0, 1, self.BASE_HEIGHT + 0.2], [0, 0, 1]])
        fk = self._fk.numpy(
            q_current[1:],
            child_link=i-1,
            position_only=False,
        )
        return np.dot(fk_base, fk)

    def render_link(self, i):
        fk = self.compute_fk(i)
        l, r, t, b = 0, self.LINK_LENGTH, 0.01, -0.01
        corner_points = [[l, b, 1], [l, t, 1], [r, t, 1], [r, b, 1]]
        transformed_corner_points = []
        for corner_point in corner_points:
            transformed_corner_points.append(
                np.dot(fk, corner_point)[0:2]
            )
        self.render_polygone(transformed_corner_points, color=(0, 0, 0))
        self.render_point(fk[0:2, 2], color=self._joint_color)

    def render_end_effector(self):
        fk = self.compute_fk(self.n)
        self.render_point(fk[0:2, 2], color=self._joint_color)
