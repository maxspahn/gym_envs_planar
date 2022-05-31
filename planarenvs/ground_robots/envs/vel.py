import numpy as np
from gym import spaces

from planarenvs.ground_robots.envs.ground_robot_env import GroundRobotEnv


class GroundRobotVelEnv(GroundRobotEnv):
    def set_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-self._limUpPos, high=self._limUpPos, dtype=np.float32
                ),
                "xdot": spaces.Box(
                    low=-self._limUpVel, high=self._limUpVel, dtype=np.float32
                ),
                "vel": spaces.Box(
                    low=-self._limUpRelVel,
                    high=self._limUpRelVel,
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-self._limUpRelVel, high=self._limUpRelVel, dtype=np.float32
        )

    def integrate(self):
        super().integrate()
        self._state["vel"] = self._action
        self._state["xdot"] = self.compute_xdot(
            self._state["x"], self._state["vel"]
        )

    def continuous_dynamics(self, x, t):
        # x = [x, y, theta, vel_for, vel_rel, xdot, ydot, thetadot]
        x_pos = x[0:3]
        xdot = self.compute_xdot(x_pos, self._action)
        veldot = np.zeros(2)
        xddot = np.zeros(3)
        return np.concatenate((xdot, veldot, xddot))
