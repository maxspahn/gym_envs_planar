import numpy as np
from gym import spaces

from planarenvs.ground_robots.envs.ground_robot_arm_env import GroundRobotArmEnv


class GroundRobotArmVelEnv(GroundRobotArmEnv):
    def set_spaces(self):
        self.observation_space = spaces.Dict(
                {"joint_state": spaces.Dict({
                    "position": spaces.Box(
                        low=np.concatenate((-self._lim_up_pos, -self._lim_up_arm_pos)),
                        high=np.concatenate((self._lim_up_pos, self._lim_up_arm_pos)),
                        dtype=np.float64
                ),
                    "velocity": spaces.Box(
                        low=np.concatenate((-self._lim_up_vel, -self._lim_up_arm_vel)),
                        high=np.concatenate((self._lim_up_vel, self._lim_up_arm_vel)),
                        dtype=np.float64
                ),
                "forward_velocity": spaces.Box(
                    low=-self._lim_up_rel_vel,
                    high=self._lim_up_rel_vel,
                    dtype=np.float64,
                ),
            })
            }
        )
        a = np.concatenate((self._lim_up_rel_vel, self._lim_up_arm_vel))
        self.action_space = spaces.Box(low=-a, high=a, dtype=np.float64)

    def integrate(self):
        super().integrate()
        qdot = self._action[2:]
        self._state["joint_state"]["forward_velocity"] = self._action[0:2]
        xdot_base = self.compute_xdot(
            self._state["joint_state"]["position"][0:3], self._state["joint_state"]["forward_velocity"]
        )
        self._state["joint_state"]["velocity"] = np.concatenate((xdot_base, qdot))

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, q, vel_rel, qdot]
        qdot = self._action[2:]
        x_base_pos = x[0:3]
        xdot_base = self.compute_xdot(x_base_pos, self._action[0:2])
        veldot = np.zeros(2)
        xddot = np.zeros(3)
        qddot = np.zeros(self._n_arm)
        return np.concatenate((xdot_base, veldot, xddot, qdot, qddot))
