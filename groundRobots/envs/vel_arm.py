import numpy as np
from gym import spaces

from groundRobots.envs.groundRobotArmEnv import GroundRobotArmEnv


class GroundRobotArmVelEnv(GroundRobotArmEnv):
    def setSpaces(self):
        o = np.concatenate(
            (
                self._limUpPos,
                self._limUpArmPos,
                self._limUpRelVel,
                self._limUpArmVel,
                self._limUpVel,
            )
        )
        self.observation_space = spaces.Box(low=-o, high=o, dtype=np.float64)
        a = np.concatenate((self._limUpRelVel, self._limUpArmVel))
        self.action_space = spaces.Box(low=-a, high=a, dtype=np.float64)

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, q, vel_rel, qdot]
        nx = 3 + self._n_arm
        x_base = self.state[0:3]
        vel_rel = self.action[0:2]
        qdot = self.action[2 : 2 + self._n_arm]
        xdot_base = np.array(
            [np.cos(x_base[2]) * vel_rel[0], np.sin(x_base[2]) * vel_rel[0], vel_rel[1]]
        )
        xdot = np.concatenate((xdot_base, qdot))
        self.state[nx : nx + 2 + self._n_arm] = self.action
        self.pos_der = xdot_base
        xddot = np.zeros(2 + self._n_arm)
        return np.concatenate((xdot, xddot))
