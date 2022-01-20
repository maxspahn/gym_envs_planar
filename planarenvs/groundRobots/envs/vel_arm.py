import numpy as np
from gym import spaces

from planarenvs.groundRobots.envs.groundRobotArmEnv import GroundRobotArmEnv


class GroundRobotArmVelEnv(GroundRobotArmEnv):
    def setSpaces(self):
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-self._limUpPos, high=self._limUpPos, dtype=np.float64),
            'q': spaces.Box(low=-self._limUpArmPos, high=self._limUpArmPos, dtype=np.float64),
            'xdot': spaces.Box(low=-self._limUpVel, high=self._limUpVel, dtype=np.float64),
            'vel': spaces.Box(low=-self._limUpRelVel, high=self._limUpRelVel, dtype=np.float64),
            'qdot': spaces.Box(low=-self._limUpArmVel, high=self._limUpArmVel, dtype=np.float64),
        })
        a = np.concatenate((self._limUpRelVel, self._limUpArmVel))
        self.action_space = spaces.Box(low=-a, high=a, dtype=np.float64)

    def integrate(self):
        super().integrate()
        self.state['qdot'] = self.action[2:]
        self.state['vel'] = self.action[0:2]
        self.state['xdot'] = self.computeXdot(self.state['x'][0:3], self.state['vel'])

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, q, vel_rel, qdot]
        qdot = self.action[2:]
        x_base_pos = x[0:3]
        xdot_base = self.computeXdot(x_base_pos, self.action[0:2])
        veldot = np.zeros(2)
        xddot = np.zeros(3)
        qddot = np.zeros(self._n_arm)
        return np.concatenate((xdot_base, veldot, xddot, qdot, qddot))
