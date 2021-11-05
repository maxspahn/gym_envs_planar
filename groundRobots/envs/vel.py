import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding


class GroundRobotVelEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    BASE_WIDTH = 1.0 # [m]
    BASE_LENGTH = 1.3 # [m]
    BASE_WHEEL_DIST = 0.6 # [m]
    LINK_MASS_BASE = 5.0  #: [kg] mass of link 1

    MAX_POS_BASE_X = 5
    MAX_POS_BASE_Y = 5
    MAX_POS_BASE_THETA = np.pi
    MAX_VEL_BASE_X = 5
    MAX_VEL_BASE_Y = 5
    MAX_VEL_BASE_THETA = 5
    MAX_VEL_WHEEL = 1

    actionlimits = [np.array([-MAX_VEL_WHEEL, -MAX_VEL_WHEEL]), np.array([MAX_VEL_WHEEL, MAX_VEL_WHEEL])]


    def __init__(self, dt=0.01):
        self.viewer = None
        high = np.array(
            [
                self.MAX_POS_BASE_X,
                self.MAX_POS_BASE_Y,
                self.MAX_POS_BASE_THETA,
                self.MAX_VEL_BASE_X,
                self.MAX_VEL_BASE_Y,
                self.MAX_VEL_BASE_THETA,
            ],
            dtype=np.float32,
        )
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=self.actionlimits[0], high=self.actionlimits[1], dtype=np.float64
        )
        self.state = None
        self._dt = dt
        self.seed()

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, pos=None, vel=None):
        try:
            if pos is None:
                pos = np.ones(3) * 0.0
            if vel is None:
                vel = np.ones(2) * 0.0
        except:
            print("Using initial data")
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        ns = self.integrate()
        vels = self.continuous_dynamics(self.state[0:3], 0.0)
        self.state = np.append(ns, vels)
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        return self.state

    def _terminal(self):
        return False

    def continuous_dynamics(self, x, t):
        vr = self.action[0]
        vl = self.action[1]
        theta = self.state[2]
        xdot = 0.5 * (vr + vl) * np.cos(theta)
        ydot = 0.5 * (vr + vl) * np.sin(theta)
        thetadot = 0.5 * (vr - vl) * self.BASE_WHEEL_DIST
        return np.array([xdot, ydot, thetadot])

    def integrate(self):
        x0 = self.state[0:3]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        bound_x = self.MAX_POS_BASE_X + 1.0
        bound_y = self.MAX_POS_BASE_Y + 1.0
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bound_x, bound_x, -bound_y, bound_y)
        self.viewer.draw_line((-bound_x, 0.0), (bound_x, 0.0))
        self.viewer.draw_line((0.0, -bound_y), (0.0, bound_y))

        if s is None:
            return None

        p = [s[0], s[1]]

        theta = s[2]
        tf = rendering.Transform(rotation=theta, translation=p)

        l, r, t, b = -0.5*self.BASE_LENGTH, 0.5 * self.BASE_LENGTH, 0.5 * self.BASE_WIDTH, -0.5 * self.BASE_WIDTH
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        yw = self.BASE_WHEEL_DIST/2.0
        wheelfl = self.viewer.draw_polygon([(0.2, yw), (0.2, yw+0.1), (0.4, yw+0.1), (0.4, yw)])
        wheelfr = self.viewer.draw_polygon([(0.2, -yw), (0.2, -yw-0.1), (0.4, -yw-0.1), (0.4, -yw)])
        wheelbl = self.viewer.draw_polygon([(-0.2, yw), (-0.2, yw+0.1), (-0.4, yw+0.1), (-0.4, yw)])
        wheelbr = self.viewer.draw_polygon([(-0.2, -yw), (-0.2, -yw-0.1), (-0.4, -yw-0.1), (-0.4, -yw)])
        wheelfl.add_attr(tf)
        wheelfr.add_attr(tf)
        wheelbl.add_attr(tf)
        wheelbr.add_attr(tf)
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
