import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding


class MobileRobotVelEnv(core.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    BASE_HEIGHT = 1.0  # [m]
    LINK_LENGTH = 1.0  # [m]

    MAX_VEL_BASE = 1
    MAX_POS_BASE = 5.0
    MAX_VEL = 4 * pi
    MAX_POS = pi

    def __init__(self, render=False, n=2, dt=0.01):
        self.viewer = None
        self._n = n
        limUpBasePos = [self.MAX_POS_BASE]
        limUpBaseVel = [self.MAX_VEL_BASE]
        limUpPos = [self.MAX_POS for i in range(n)]
        limUpVel = [self.MAX_VEL for i in range(n)]
        high = np.array(limUpBasePos + limUpPos + limUpBaseVel + limUpVel, dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.array(limUpBaseVel + limUpVel), high=np.array(limUpBaseVel + limUpVel), dtype=np.float64
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
        pos = [0.1 for i in range(self._n)]
        vel = [0.1 for i in range(self._n)]
        basePos = [0.0]
        baseVel = [0.0]
        self.state = np.array(basePos + pos + baseVel + vel)
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        ns = self.integrate()
        self.state = np.concatenate((ns, a))
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
        x0 = self.state[0:1 + self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def forwardKinematics(self, lastLinkIndex):
        fk = np.array([self.state[0], 1.2, 0.0])
        for i in range(lastLinkIndex):
            angle = 0.0
            for j in range(i + 1):
                angle += self.state[j + 1]
            fk[0] += np.cos(angle) * self.LINK_LENGTH
            fk[1] += np.sin(angle) * self.LINK_LENGTH
            fk[2] += self.state[i + 1]
        fk[2] += self.state[lastLinkIndex + 1]
        return fk

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
        base = self.viewer.draw_polygon([(-0.2, -0.2), (0.0, 0.0), (0.0, 0.0), (0.2, -0.2)])
        baseJoint = self.viewer.draw_circle(.10)
        baseJoint.set_color(.8, .8, 0)
        tf0 = rendering.Transform(rotation=0, translation=(self.state[0], 1.2))
        baseJoint.add_attr(tf0)
        base.add_attr(tf0)
        l, r, t, b = 0, self.LINK_LENGTH, .01, -.01
        for i in range(self._n):
            fk = self.forwardKinematics(i)
            tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.set_color(0, .8, .8)
            link.add_attr(tf)
            joint = self.viewer.draw_circle(.10)
            joint.set_color(.8, .8, 0)
            joint.add_attr(tf)
        fk = self.forwardKinematics(self._n)
        tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
        eejoint = self.viewer.draw_circle(.10)
        eejoint.set_color(.8, .8, 0)
        eejoint.add_attr(tf)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
