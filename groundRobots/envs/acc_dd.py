import numpy as np
from numpy import sin, cos, pi

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding


class GroundRobotDiffDriveAccEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    BASE_WIDTH = 1.0 # [m]
    BASE_LENGTH = 1.3 # [m]
    BASE_WHEEL_DIST = 0.6 # [m]
    LINK_MASS_BASE = 5.0  #: [kg] mass of link 1

    MAX_POS_BASE_X = 10
    MAX_POS_BASE_Y = 10
    MAX_POS_BASE_THETA = np.pi
    MAX_VEL_BASE_X = 5
    MAX_VEL_BASE_Y = 5
    MAX_VEL_BASE_THETA = 5
    MAX_VEL_WHEEL = 1
    MAX_FORWARD_VEL = 1.0
    MAX_ROTATION_VEL = 1.0
    MAX_FORWARD_ACC = 1.0
    MAX_ROTATION_ACC = 1.0

    actionlimits = [np.array([-MAX_FORWARD_ACC, -MAX_ROTATION_ACC]), np.array([MAX_FORWARD_ACC, MAX_ROTATION_ACC])]


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
                self.MAX_FORWARD_VEL,
                self.MAX_ROTATION_VEL
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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, initPos, initForwardVel):
        initVel = np.array([np.cos(initPos[2]) * initForwardVel[0],
                            np.sin(initPos[2]) * initForwardVel[0],
                            initForwardVel[1]])
        self.state = np.concatenate((initPos, initVel, initForwardVel))
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
        return False

    def continuous_dynamics(self, x, t):
        a_forward = self.action[0]
        a_rotation = self.action[1]
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        xdot = self.state[3]
        ydot = self.state[4]
        thetadot = self.state[5]
        v_forward = self.state[6]
        v_rotation = self.state[7]
        xddot = (a_forward) * np.cos(theta) - v_forward * np.sin(theta) * thetadot
        yddot = (a_forward) * np.sin(theta) + v_forward * np.cos(theta) * thetadot
        thetaddot = a_rotation
        return np.array([xdot, ydot, thetadot, xddot, yddot, thetaddot, a_forward, a_rotation])

    def integrate(self):
        x0 = self.state
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        bound_x = self.MAX_POS_BASE_X + 1.0
        bound_y = self.MAX_POS_BASE_Y + 1.0
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 1000)
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

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
