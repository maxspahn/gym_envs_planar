import numpy as np
from gym import spaces

from planarenvs.ground_robots.envs.ground_robot_arm_env import GroundRobotArmEnv


class GroundRobotArmVelEnv(GroundRobotArmEnv):
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
        a = np.concatenate((self._lim_up_rel_vel, self._lim_up_arm_vel))
        self.action_space = spaces.Box(low=-a, high=a, dtype=np.float64)

    def integrate(self):
        super().integrate()
        self._state["qdot"] = self._action[2:]
        self._state["vel"] = self._action[0:2]
        self._state["xdot"] = self.compute_xdot(
            self._state["x"][0:3], self._state["vel"]
        )

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, q, vel_rel, qdot]
        qdot = self._action[2:]
        x_base_pos = x[0:3]
        xdot_base = self.compute_xdot(x_base_pos, self._action[0:2])
        veldot = np.zeros(2)
        xddot = np.zeros(3)
        qddot = np.zeros(self._n_arm)
        return np.concatenate((xdot_base, veldot, xddot, qdot, qddot))
