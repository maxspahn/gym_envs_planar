import numpy as np
import time
from scipy.integrate import odeint

from groundRobots.envs.groundRobotEnv import GroundRobotEnv


class GroundRobotArmEnv(GroundRobotEnv):
    LINK_LENGTH = 1.0
    MAX_ARM_VEL = 4 * np.pi
    MAX_ARM_POS = 5 * np.pi
    MAX_ARM_ACC = 9 * np.pi

    def __init__(self, render=False, dt=0.01, n_arm=1):
        self._n_arm = n_arm
        self._limUpArmPos = np.ones(self._n_arm) * self.MAX_ARM_POS
        self._limUpArmVel = np.ones(self._n_arm) * self.MAX_ARM_VEL
        self._limUpArmAcc = np.ones(self._n_arm) * self.MAX_ARM_ACC
        super().__init__(render=render, dt=dt)

    def integrate(self):
        self._t += self.dt()
        x0 = np.concatenate((self.state['x'], self.state['vel'], self.state['xdot'], self.state['q'], self.state['qdot']))
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        self.state['x'] = ynext[1][0:3]
        self.state['vel'] = ynext[1][3:5]
        self.state['xdot'] = ynext[1][5:8]
        self.state['q'] = ynext[1][8:8+self._n_arm]
        self.state['qdot'] = ynext[1][8+self._n_arm:]

    def reset(self, pos=None, vel=None):
        self.resetCommon()
        """ The velocity is the forward velocity and turning velocity here """
        if not isinstance(pos, np.ndarray) or not pos.size == 3 + self._n_arm:
            pos = np.zeros(3 + self._n_arm)
        if not isinstance(vel, np.ndarray) or not vel.size == 2 + self._n_arm:
            vel = np.zeros(2 + self._n_arm)
        xdot_base = self.computeXdot(pos[0:3], vel[0:2])
        self.state = {'x': pos[0:3], 'vel': vel[0:2], 'xdot': xdot_base, 'q': pos[3:], 'qdot': vel[2:]}
        return self._get_ob()

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering
        super().render(mode=mode, final=False)

        # arm
        p = self.state['x'][0:2]
        theta = self.state['x'][2]
        q = self.state['q']
        l, r, t, b = 0, self.LINK_LENGTH, 0.05, -0.05
        p_arm = p + 0.2 * np.array([np.cos(theta), np.sin(theta)])
        tf_arm = rendering.Transform(rotation=theta + q, translation=p_arm)
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.2, 0.8)
        link.add_attr(tf_arm)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
