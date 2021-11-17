import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding

from groundRobots.envs.groundRobotEnv import GroundRobotEnv


class GroundRobotAccEnv(GroundRobotEnv):
    def setSpaces(self):
        o = np.concatenate((self._limUpPos, self._limUpRelVel, self._limUpVel))
        self.observation_space = spaces.Box(low=-o, high=o, dtype=np.float64)
        self.action_space = spaces.Box(
            low=-self._limUpRelAcc, high=self._limUpRelAcc, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, vel_rel]
        x = self.state[0:3]
        vel_rel = self.state[3:5]
        xdot = np.array(
            [np.cos(x[2]) * vel_rel[0], np.sin(x[2]) * vel_rel[0], vel_rel[1]]
        )
        self.pos_der = xdot
        xddot = self.action
        return np.concatenate((xdot, xddot))
