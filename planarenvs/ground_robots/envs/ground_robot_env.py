import numpy as np
import time
from scipy.integrate import odeint
from abc import abstractmethod

from planarenvs.planar_common.planar_env import PlanarEnv


class GroundRobotEnv(PlanarEnv):

    BASE_WIDTH = 1.0  # [m]
    BASE_LENGTH = 1.3  # [m]
    BASE_WHEEL_DIST = 0.6  # [m]
    LINK_MASS_BASE = 5.0  #: [kg] mass of link 1

    MAX_POS_BASE = 5
    MAX_POS_BASE_THETA = 5 * np.pi
    MAX_VEL_BASE = 5
    MAX_VEL_BASE_THETA = 5
    MAX_ACC_BASE = 100
    MAX_ACC_BASE_THETA = 100
    MAX_VEL_FORWARD = 3.0
    MAX_ACC_FORWARD = 100

    def __init__(self, render=False, dt=0.01):
        super().__init__(render=render, dt=dt)
        self._lim_up_pos = np.array(
            [self.MAX_POS_BASE, self.MAX_POS_BASE, self.MAX_POS_BASE_THETA]
        )
        self._lim_up_vel = np.array(
            [self.MAX_VEL_BASE, self.MAX_VEL_BASE, self.MAX_VEL_BASE_THETA]
        )
        self._lim_up_rel_vel = np.array(
            [self.MAX_VEL_FORWARD, self.MAX_VEL_BASE_THETA]
        )
        self._lim_up_acc = np.array(
            [self.MAX_ACC_BASE, self.MAX_ACC_BASE, self.MAX_ACC_BASE_THETA]
        )
        self._lim_up_rel_acc = np.array(
            [self.MAX_ACC_FORWARD, self.MAX_ACC_BASE_THETA]
        )
        self.set_spaces()

    @abstractmethod
    def set_spaces(self):
        pass

    def reset(self, pos=None, vel=None):
        self.reset_common()
        # The velocity is the forward velocity and turning velocity here.
        if not isinstance(pos, np.ndarray) or not pos.size == 3:
            pos = np.zeros(3)
        if not isinstance(vel, np.ndarray) or not vel.size == 2:
            vel = np.zeros(2)
        self._state = {
            "x": pos,
            "vel": vel,
            "xdot": self.compute_xdot(pos, vel),
        }
        self._sensor_state = {}
        return self._get_ob()

    def compute_xdot(self, x, vel):
        assert x.size == 3
        assert vel.size == 2
        return np.array(
            [
                np.cos(x[2]) * vel[0],
                np.sin(x[2]) * vel[0],
                vel[1],
            ]
        )

    def _reward(self):
        reward = -1.0 if not self._terminal() else 0.0
        return reward

    def integrate(self):
        self._t += self.dt()
        x0 = np.concatenate(
            (self._state["x"], self._state["vel"], self._state["xdot"])
        )
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        self._state["x"] = ynext[1][0:3]
        self._state["vel"] = ynext[1][3:5]
        self._state["xdot"] = ynext[1][5:8]

    def _terminal(self):
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human", final=True):
        bound_x = self.MAX_POS_BASE + 1.0
        bound_y = self.MAX_POS_BASE + 1.0
        bounds = [bound_x, bound_y]
        self.render_common(bounds)
        from gym.envs.classic_control import rendering #pylint: disable=import-outside-toplevel

        # drawAxis
        self._viewer.draw_line((-bound_x, 0.0), (bound_x, 0.0))
        self._viewer.draw_line((0.0, -bound_y), (0.0, bound_y))

        p = self._state["x"][0:2]
        theta = self._state["x"][2]

        tf = rendering.Transform(rotation=theta, translation=p)

        l, r, t, b = (
            -0.5 * self.BASE_LENGTH,
            0.5 * self.BASE_LENGTH,
            0.5 * self.BASE_WIDTH,
            -0.5 * self.BASE_WIDTH,
        )
        link = self._viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        yw = self.BASE_WHEEL_DIST / 2.0
        wheelfl = self._viewer.draw_polygon(
            [(0.2, yw), (0.2, yw + 0.1), (0.4, yw + 0.1), (0.4, yw)]
        )
        wheelfr = self._viewer.draw_polygon(
            [(0.2, -yw), (0.2, -yw - 0.1), (0.4, -yw - 0.1), (0.4, -yw)]
        )
        wheelbl = self._viewer.draw_polygon(
            [(-0.2, yw), (-0.2, yw + 0.1), (-0.4, yw + 0.1), (-0.4, yw)]
        )
        wheelbr = self._viewer.draw_polygon(
            [(-0.2, -yw), (-0.2, -yw - 0.1), (-0.4, -yw - 0.1), (-0.4, -yw)]
        )
        wheelfl.add_attr(tf)
        wheelfr.add_attr(tf)
        wheelbl.add_attr(tf)
        wheelbr.add_attr(tf)
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)

        if final:
            time.sleep(self.dt())
            return self._viewer.render(return_rgb_array=mode == "rgb_array")
