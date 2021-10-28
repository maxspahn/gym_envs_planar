import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding

from nLinkReacher.resources.createDynamics import createDynamics


class NLinkTorReacherEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    LINK_LENGTH = 1.0  # [m]
    LINK_MASS = 1.0

    MAX_VEL = 4 * pi
    MAX_POS = pi
    MAX_ACC = 9 * pi
    MAX_TOR = 1000

    def __init__(self, n=2, dt=0.01, k=0.0):
        self._n = n
        self._k = k
        self.viewer = None
        self.dynamics_fun, _, _ = createDynamics(n)
        limUpPos = [self.MAX_POS for i in range(n)]
        limUpVel = [self.MAX_VEL for i in range(n)]
        limUpAcc = [self.MAX_ACC for i in range(n)]
        limUpTor = [self.MAX_TOR for i in range(n)]
        high = np.array( limUpPos + limUpVel + limUpAcc, dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.array(limUpTor), high=np.array(limUpTor), dtype=np.float64
        )
        self.state = None
        self.seed()
        self._dt = dt

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        pos = [0 for i in range(self._n)]
        vel = [0 for i in range(self._n)]
        acc = [0 for i in range(self._n)]
        self.state = np.array(pos + vel + acc)
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        ns = self.integrate()
        acc = self.continuous_dynamics(self.state[0:self._n * 2], 0.1)
        self.state = np.concatenate((ns, acc[0:self._n]))
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        return self.state

    def _terminal(self):
        s = self.state
        return False

    def continuous_dynamics(self, x, t):
        q = x[0:self._n]
        qdot = x[self._n: self._n * 2]
        l = [self.LINK_LENGTH for i in range(self._n)]
        m = [self.LINK_MASS for i in range(self._n)]
        g = 10.0
        tau = self.action
        acc = np.array(self.dynamics_fun(q, qdot, l, m, g, self._k, tau))[:, 0]
        return acc

    def integrate(self):
        x0 = self.state[0:2 * self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def forwardKinematics(self, lastLinkIndex):
        fk = np.array([0.0, 0.2, 0.0])
        for i in range(lastLinkIndex):
            angle = 0.0
            for j in range(i+1):
                angle += self.state[j]
            fk[0] += np.cos(angle) * self.LINK_LENGTH
            fk[1] += np.sin(angle) * self.LINK_LENGTH
            fk[2] += self.state[i]
        fk[2] += self.state[lastLinkIndex]
        return fk

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        bound = self.LINK_LENGTH * self._n + 0.2
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        self.viewer.draw_line((-bound, 0), (bound, 0))
        base = self.viewer.draw_polygon([(-0.2,0), (0.0,0.2), (0.2,0), (-0.2,0)])
        baseJoint = self.viewer.draw_circle(.10)
        baseJoint.set_color(.8, .8, 0)
        tf0 = rendering.Transform(rotation=0, translation=(0.0, 0.2))
        baseJoint.add_attr(tf0)
        l,r,t,b = 0, self.LINK_LENGTH, .01, -.01
        for i in range(self._n):
            fk = self.forwardKinematics(i)
            tf = rendering.Transform(rotation=fk[2], translation=fk[0:2])
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.set_color(0,.8, .8)
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
