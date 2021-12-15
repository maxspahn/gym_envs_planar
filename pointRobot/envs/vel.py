import numpy as np
from gym import spaces

from pointRobot.envs.pointRobotEnv import PointRobotEnv


class PointRobotVelEnv(PointRobotEnv):
    def setSpaces(self):
        o = np.concatenate((self._limUpPos, self._limUpVel))
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-self._limUpPos, high=self._limUpPos, dtype=np.float64),
            'xdot': spaces.Box(low=-self._limUpVel, high=self._limUpVel, dtype=np.float64)
        })
        self.action_space = spaces.Box(
            low=-self._limUpVel, high=self._limUpVel, dtype=np.float64
        )

    def integrate(self):
        super().integrate()
        self.state['xdot'] = self.action

    def continuous_dynamics(self, x, t):
        vel = self.action
        acc = np.zeros(self._n)
        xdot = np.concatenate((vel, acc))
        return xdot
