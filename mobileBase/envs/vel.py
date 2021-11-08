import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding


class MobileBaseVelEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    BASE_HEIGHT = 1.0 # [m]
    LINK_MASS_BASE = 500.0  #: [kg] mass of link 1

    MAX_VEL_BASE = 1
    MAX_POS_BASE = 5.0

    actionlimits = [np.array([-MAX_VEL_BASE]), np.array([MAX_VEL_BASE])]


    def __init__(self, render=False, dt=0.01):
        self.viewer = None
        high = np.array(
            [
                self.MAX_POS_BASE,
                self.MAX_VEL_BASE
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
        self._render = render


    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.zeros(shape=(2))
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        ns = self.integrate()
        self.state = np.append(ns, a)
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        return self.state

    def _terminal(self):
        if self.state[0] > self.MAX_POS_BASE or self.state[0] < -self.MAX_POS_BASE:
            return True
        return False

    def continuous_dynamics(self, x, t):
        return self.action

    def integrate(self):
        x0 = self.state[0]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.MAX_POS_BASE + 1.0
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p0 = [s[0], 0.5 * self.BASE_HEIGHT]
        p1 = [p0[0], p0[1] + 0.5 * self.BASE_HEIGHT]

        p = [p0, p1]
        thetas = [0.0, 0.0]
        tf0 = rendering.Transform(rotation=thetas[0], translation=p0)
        tf1 = rendering.Transform(rotation=thetas[1], translation=p1)
        tf = [tf0, tf1]

        self.viewer.draw_line((-5.5, 0), (5.5, 0))

        l, r, t, b = -0.5, 0.5, 0.5, -0.5
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf[0])
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
