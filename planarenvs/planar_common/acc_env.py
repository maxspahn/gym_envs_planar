import numpy as np
from gym import spaces


class AccEnv(object):
    def set_spaces(self):
        self.observation_space = spaces.Dict(
            {"joint_state": spaces.Dict({
                "position": spaces.Box(
                    low=-self._lim_up_pos, high=self._lim_up_pos, dtype=np.float64
                ),
                "velocity": spaces.Box(
                    low=-self._lim_up_vel, high=self._lim_up_vel, dtype=np.float64
                ),
            })}
        )
        self.action_space = spaces.Box(
            low=-self._lim_up_acc, high=self._lim_up_acc, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):  # pylint: disable=unused-argument
        vel = x[self._n : self._n * 2]
        xdot = np.concatenate((vel, self._action))
        return xdot
