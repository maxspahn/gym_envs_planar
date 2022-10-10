import numpy as np
from gym import spaces

from planarenvs.ground_robots.envs.ground_robot_env import GroundRobotEnv


class GroundRobotVelEnv(GroundRobotEnv):
    def set_spaces(self):
        self.observation_space = spaces.Dict(
            {"joint_state": spaces.Dict({
                "position": spaces.Box(
                    low=-self._lim_up_pos,
                    high=self._lim_up_pos,
                    dtype=np.float64
                ),
                "velocity": spaces.Box(
                    low=-self._lim_up_vel,
                    high=self._lim_up_vel,
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
        self.action_space = spaces.Box(
            low=-self._lim_up_rel_vel,
            high=self._lim_up_rel_vel,
            dtype=np.float64
        )

    def integrate(self):
        super().integrate()
        self._state["joint_state"]["forward_velocity"] = self._action
        self._state["joint_state"]["velocity"] = self.compute_xdot(
            self._state["joint_state"]["position"], self._state["joint_state"]["forward_velocity"]
        )

    def continuous_dynamics(self, x, t):
        # x = [x, y, theta, vel_for, vel_rel, xdot, ydot, thetadot]
        x_pos = x[0:3]
        xdot = self.compute_xdot(x_pos, self._action)
        veldot = np.zeros(2)
        xddot = np.zeros(3)
        return np.concatenate((xdot, veldot, xddot))
