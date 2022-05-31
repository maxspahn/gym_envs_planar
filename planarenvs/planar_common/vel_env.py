import numpy as np
from gym import spaces


class VelEnv(object):
    def set_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-self._lim_up_pos, high=self._lim_up_pos, dtype=np.float64
                ),
                "xdot": spaces.Box(
                    low=-self._lim_up_vel, high=self._lim_up_vel, dtype=np.float64
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-self._lim_up_vel, high=self._lim_up_vel, dtype=np.float64
        )

    def integrate(self):
        super().integrate()
        self._state["xdot"] = self._action

    def continuous_dynamics(self, x, t):  # pylint: disable=unused-argument
        vel = self._action
        acc = np.zeros(self._n)
        xdot = np.concatenate((vel, acc))
        return xdot
