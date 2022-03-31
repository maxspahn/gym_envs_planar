import numpy as np
from gym import spaces

from planarenvs.ground_robots.envs.ground_robot_arm_env import GroundRobotArmEnv


class GroundRobotArmAccEnv(GroundRobotArmEnv):
    def set_spaces(self):
        self._observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-self._limUpPos, high=self._limUpPos, dtype=np.float64
                ),
                "q": spaces.Box(
                    low=-self._limUpArmPos,
                    high=self._limUpArmPos,
                    dtype=np.float64,
                ),
                "xdot": spaces.Box(
                    low=-self._limUpVel, high=self._limUpVel, dtype=np.float64
                ),
                "vel": spaces.Box(
                    low=-self._limUpRelVel,
                    high=self._limUpRelVel,
                    dtype=np.float64,
                ),
                "qdot": spaces.Box(
                    low=-self._limUpArmVel,
                    high=self._limUpArmVel,
                    dtype=np.float64,
                ),
            }
        )
        a = np.concatenate((self._limUpRelAcc, self._limUpArmAcc))
        self.action_space = spaces.Box(low=-a, high=a, dtype=np.float64)

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, vel, xdot_base, q, qdot)
        x_pos = x[0:3]
        vel = x[3:5]
        xdot_base = self.compute_xdot(x_pos, vel)
        veldot = self._action[0:2]
        qddot = self._action[2:]
        qdot = x[8 + self._n_arm :]
        xddot = np.zeros(3)
        return np.concatenate((xdot_base, veldot, xddot, qdot, qddot))
