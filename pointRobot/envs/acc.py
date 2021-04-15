import numpy as np
from numpy import sin, cos, pi

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding


class PointRobotAccEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    MAX_VEL = 10
    MAX_POS = 10
    MAX_ACC = 10

    def __init__(self, n=2, dt=0.01):
        self._n = n
        self.viewer = None
        limUpPos = [self.MAX_POS for i in range(n)]
        limUpVel = [self.MAX_VEL for i in range(n)]
        limUpAcc = [self.MAX_ACC for i in range(n)]
        high = np.array(limUpPos + limUpVel,  dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-np.array(limUpAcc), high=np.array(limUpAcc), dtype=np.float64
        )
        self.state = None
        self.seed()
        self._dt = dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, initPos, initVel):
        self.state = np.concatenate((initPos, initVel))
        return self._get_ob()

    def step(self, a):
        s = self.state
        self.action = a
        ns = self.integrate()
        self.state = ns
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        return self.state

    def _terminal(self):
        s = self.state
        return False

    def continuous_dynamics(self, x, t):
        u = self.action
        vel = np.array(x[self._n : self._n * 2])
        acc = np.concatenate((vel, u))
        return acc

    def integrate(self):
        x0 = self.state[0:2 * self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state
        if s is None:
            return None

        bound = 5.0
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        self.viewer.draw_line((-bound, 0), (bound, 0))
        self.viewer.draw_line((0, -bound), (0, bound))
        x = s[0]
        y = 0.0
        if self._n == 2:
            y = s[1]
        tf0 = rendering.Transform(rotation=0, translation=(x, y))
        joint = self.viewer.draw_circle(.10)
        joint.set_color(.8, .8, 0)
        joint.add_attr(tf0)


        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
