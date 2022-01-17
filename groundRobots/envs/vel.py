import numpy as np
from gym import spaces

from groundRobots.envs.groundRobotEnv import GroundRobotEnv


class GroundRobotVelEnv(GroundRobotEnv):
    def setSpaces(self):
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-self._limUpPos, high=self._limUpPos, dtype=np.float64),
            'xdot': spaces.Box(low=-self._limUpVel, high=self._limUpVel, dtype=np.float64),
            'vel': spaces.Box(low=-self._limUpRelVel, high=self._limUpRelVel, dtype=np.float64),
        })
        self.action_space = spaces.Box(
            low=-self._limUpRelVel, high=self._limUpRelVel, dtype=np.float64
        )

    def integrate(self):
        super().integrate()
        self.state['vel'] = self.action
        self.state['xdot'] = self.computeXdot(self.state['x'], self.state['vel'])

    def continuous_dynamics(self, x, t):
        # x = [x, y, theta, vel_for, vel_rel, xdot, ydot, thetadot]
        x_pos = x[0:3]
        xdot = self.computeXdot(x_pos, self.action)
        veldot = np.zeros(2)
        xddot = np.zeros(3)
        return np.concatenate((xdot, veldot, xddot))
