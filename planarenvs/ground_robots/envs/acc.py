import numpy as np
from gym import spaces

from planarenvs.ground_robots.envs.ground_robot_env import GroundRobotEnv


class GroundRobotAccEnv(GroundRobotEnv):
    def set_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-self._lim_up_pos,
                    high=self._lim_up_pos,
                    dtype=np.float64
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
            }
        )
        self.action_space = spaces.Box(
            low=-self._lim_up_rel_acc,
            high=self._lim_up_rel_acc,
            dtype=np.float64
        )

    def integrate(self):
        super().integrate()
        self._state["xdot"] = self.compute_xdot(
            self._state["x"], self._state["vel"]
        )

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, vel_rel, xdot, ydot, thetadot]
        x_pos = x[0:3]
        vel = x[3:5]
        xdot = self.compute_xdot(x_pos, vel)
        veldot = self._action
        xddot = np.zeros(3)
        return np.concatenate((xdot, veldot, xddot))
