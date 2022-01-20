import numpy as np
import time
from scipy.integrate import odeint
from abc import abstractmethod

from planarenvs.planarCommon.planarEnv import PlanarEnv


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
        self._limUpPos = np.array(
            [self.MAX_POS_BASE, self.MAX_POS_BASE, self.MAX_POS_BASE_THETA]
        )
        self._limUpVel = np.array(
            [self.MAX_VEL_BASE, self.MAX_VEL_BASE, self.MAX_VEL_BASE_THETA]
        )
        self._limUpRelVel = np.array([self.MAX_VEL_FORWARD, self.MAX_VEL_BASE_THETA])
        self._limUpAcc = np.array(
            [self.MAX_ACC_BASE, self.MAX_ACC_BASE, self.MAX_ACC_BASE_THETA]
        )
        self._limUpRelAcc = np.array([self.MAX_ACC_FORWARD, self.MAX_ACC_BASE_THETA])
        self.setSpaces()

    @abstractmethod
    def setSpaces(self):
        pass

    def reset(self, pos=None, vel=None):
        self.resetCommon()
        """ The velocity is the forward velocity and turning velocity here """
        if not isinstance(pos, np.ndarray) or not pos.size == 3:
            pos = np.zeros(3)
        if not isinstance(vel, np.ndarray) or not vel.size == 2:
            vel = np.zeros(2)
        self.state = {'x': pos, 'vel': vel, 'xdot': self.computeXdot(pos, vel)}
        return self._get_ob()

    def computeXdot(self, x, vel):
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
        x0 = np.concatenate((self.state['x'], self.state['vel'], self.state['xdot']))
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        self.state['x'] = ynext[1][0:3]
        self.state['vel'] = ynext[1][3:5]
        self.state['xdot'] = ynext[1][5:8]

    def _terminal(self):
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def render(self, mode="human", final=True):
        bound_x = self.MAX_POS_BASE + 1.0
        bound_y = self.MAX_POS_BASE + 1.0
        bounds = [bound_x, bound_y]
        self.renderCommon(bounds)
        from gym.envs.classic_control import rendering

        # drawAxis
        self.viewer.draw_line((-bound_x, 0.0), (bound_x, 0.0))
        self.viewer.draw_line((0.0, -bound_y), (0.0, bound_y))

        p = self.state['x'][0:2]
        theta = self.state['x'][2]

        tf = rendering.Transform(rotation=theta, translation=p)

        l, r, t, b = (
            -0.5 * self.BASE_LENGTH,
            0.5 * self.BASE_LENGTH,
            0.5 * self.BASE_WIDTH,
            -0.5 * self.BASE_WIDTH,
        )
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        yw = self.BASE_WHEEL_DIST / 2.0
        wheelfl = self.viewer.draw_polygon(
            [(0.2, yw), (0.2, yw + 0.1), (0.4, yw + 0.1), (0.4, yw)]
        )
        wheelfr = self.viewer.draw_polygon(
            [(0.2, -yw), (0.2, -yw - 0.1), (0.4, -yw - 0.1), (0.4, -yw)]
        )
        wheelbl = self.viewer.draw_polygon(
            [(-0.2, yw), (-0.2, yw + 0.1), (-0.4, yw + 0.1), (-0.4, yw)]
        )
        wheelbr = self.viewer.draw_polygon(
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
            return self.viewer.render(return_rgb_array=mode == "rgb_array")
