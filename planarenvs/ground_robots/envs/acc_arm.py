import numpy as np
from gym import spaces

from planarenvs.ground_robots.envs.ground_robot_arm_env import GroundRobotArmEnv


class GroundRobotArmAccEnv(GroundRobotArmEnv):
    def set_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-self._lim_up_pos,
                    high=self._lim_up_pos,
                    dtype=np.float64
                ),
                "q": spaces.Box(
                    low=-self._lim_up_arm_pos,
                    high=self._lim_up_arm_pos,
                    dtype=np.float64,
                ),
                "xdot": spaces.Box(
                    low=-self._lim_up_vel,
                    high=self._lim_up_vel,
                    dtype=np.float64
                ),
                "vel": spaces.Box(
                    low=-self._lim_up_rel_vel,
                    high=self._lim_up_rel_vel,
                    dtype=np.float64,
                ),
                "qdot": spaces.Box(
                    low=-self._lim_up_arm_vel,
                    high=self._lim_up_arm_vel,
                    dtype=np.float64,
                ),
            }
        )
        a = np.concatenate((self._lim_up_rel_acc, self._lim_up_arm_acc))
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
