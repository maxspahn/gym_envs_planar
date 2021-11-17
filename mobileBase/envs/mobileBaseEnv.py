import numpy as np
from abc import abstractmethod
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding


class MobileBaseEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    BASE_HEIGHT = 1.0 # [m]
    LINK_MASS_BASE = 500.0  #: [kg] mass of link 1

    MAX_VEL = 1
    MAX_POS = 5.0
    MAX_ACC = 1.0
    MAX_FOR = 100


    def __init__(self, render=False, dt=0.01):
        self.viewer = None
        self.state = None
        self._n = 1
        self._limUpPos = np.ones(self._n) * self.MAX_POS
        self._limUpVel = np.ones(self._n) * self.MAX_VEL
        self._limUpAcc = np.ones(self._n) * self.MAX_ACC
        self._limUpFor = np.ones(self._n) * self.MAX_FOR
        self.setSpaces()
        self._dt = dt
        self.seed()
        self._render = render

    @abstractmethod
    def setSpaces(self):
        pass


    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, pos=None, vel=None):
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    def step(self, a):
        self.action = a
        _ = self.continuous_dynamics(self.state, 0.0)
        ns = self.integrate()
        self.state = ns
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        return self.state

    def _terminal(self):
        if self.state[0] > self.MAX_POS or self.state[0] < -self.MAX_POS:
            return True
        return False

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def integrate(self):
        x0 = self.state[0 : 2 * self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.MAX_POS + 1.0
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
