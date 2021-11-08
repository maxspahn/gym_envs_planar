import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding


class MobileRobotAccEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    BASE_HEIGHT = 1.0 # [m]
    LINK_LENGTH = 1.0  # [m]

    MAX_VEL_BASE = 1
    MAX_POS_BASE = 5.0
    MAX_ACC_BASE = 1.0
    MAX_POS = pi
    MAX_VEL = 4 * pi
    MAX_ACC = 4 * pi

    def __init__(self, render=False, n=2, dt=0.01):
        self.viewer = None
        self._n = n 
        limUpBasePos = [self.MAX_POS_BASE]
        limUpBaseVel = [self.MAX_VEL_BASE]
        limUpBaseAcc = [self.MAX_ACC_BASE]
        limUpPos = [self.MAX_POS for i in range(n)]
        limUpVel = [self.MAX_VEL for i in range(n)]
        limUpAcc = [self.MAX_ACC for i in range(n)]
        high = np.array(limUpBasePos + limUpPos + limUpBaseVel + limUpVel + limUpBaseAcc + limUpAcc, dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.array(limUpBaseAcc + limUpAcc), high=np.array(limUpBaseAcc + limUpAcc), dtype=np.float64
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

    def reset(self, pos=None, vel=None):
        try:
            if pos==None:
                pos=np.concatenate((np.zeros(1), np.ones(self._n) * 0.1))
            if vel==None:
                vel=np.concatenate((np.zeros(1), np.ones(self._n) * 0.1))
        except:
            print("Using initial data")
        self.state = np.concatenate((pos, vel))
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
        u = self.action
        vel = np.array(x[self._n + 1 : (self._n + 1) * 2])
        acc = np.concatenate((vel, u))
        return acc

    def integrate(self):
        x0 = self.state[0:(1 + self._n)  * 2]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def forwardKinematics(self, x_base, lastLinkIndex):
        q = self.state[0:self._n + 1]
        fk = np.zeros(3)
        if lastLinkIndex >= 1:
            fk[0] += x_base[0]
            fk[1] += x_base[1]
            fk[2] += q[1]
        for i in range(1, lastLinkIndex):
            fk[0] += np.cos(fk[2]) * self.LINK_LENGTH
            fk[1] += np.sin(fk[2]) * self.LINK_LENGTH
            if i < (self._n):
                fk[2] += q[i+1]
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
        base = self.viewer.draw_polygon([(-0.2,-0.2), (0.0,0.0), (0.0,0.0), (0.2,-0.2)])
        baseJoint = self.viewer.draw_circle(.10)
        baseJoint.set_color(.8, .8, 0)
        tf0 = rendering.Transform(rotation=0, translation=(self.state[0], 1.2))
        baseJoint.add_attr(tf0)
        base.add_attr(tf0)
        l,r,t,b = 0, self.LINK_LENGTH, .01, -.01
        x_base = np.array([self.state[0], 1.2])
        for i in range(self._n):
            fk = self.forwardKinematics(x_base, i+1)
            tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.set_color(0,.8, .8)
            link.add_attr(tf)
            joint = self.viewer.draw_circle(.10)
            joint.set_color(.8, .8, 0)
            joint.add_attr(tf)
        fk = self.forwardKinematics(x_base, self._n+1)
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
