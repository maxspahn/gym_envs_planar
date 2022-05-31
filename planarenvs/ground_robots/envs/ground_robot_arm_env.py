import numpy as np
import time
from scipy.integrate import odeint

from planarenvs.ground_robots.envs.ground_robot_env import GroundRobotEnv


class GroundRobotArmEnv(GroundRobotEnv):
    LINK_LENGTH = 1.0
    MAX_ARM_VEL = 4 * np.pi
    MAX_ARM_POS = 5 * np.pi
    MAX_ARM_ACC = 9 * np.pi

    def __init__(self, render=False, dt=0.01, n_arm=1):
        self._n_arm = n_arm
        self._lim_up_arm_pos = np.ones(self._n_arm) * self.MAX_ARM_POS
        self._lim_up_arm_vel = np.ones(self._n_arm) * self.MAX_ARM_VEL
        self._lim_up_arm_acc = np.ones(self._n_arm) * self.MAX_ARM_ACC
        super().__init__(render=render, dt=dt)

    def integrate(self):
        self._t += self.dt()
        x0 = np.concatenate(
            (
                self._state["x"],
                self._state["vel"],
                self._state["xdot"],
                self._state["q"],
                self._state["qdot"],
            )
        )
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        self._state["x"] = ynext[1][0:3]
        self._state["vel"] = ynext[1][3:5]
        self._state["xdot"] = ynext[1][5:8]
        self._state["q"] = ynext[1][8 : 8 + self._n_arm]
        self._state["qdot"] = ynext[1][8 + self._n_arm :]

    def reset(self, pos=None, vel=None):
        self.reset_common()
        # The velocity is the forward velocity and turning velocity here.
        if not isinstance(pos, np.ndarray) or not pos.size == 3 + self._n_arm:
            pos = np.zeros(3 + self._n_arm)
        if not isinstance(vel, np.ndarray) or not vel.size == 2 + self._n_arm:
            vel = np.zeros(2 + self._n_arm)
        xdot_base = self.compute_xdot(pos[0:3], vel[0:2])
        self._state = {
            "x": pos[0:3],
            "vel": vel[0:2],
            "xdot": xdot_base,
            "q": pos[3:],
            "qdot": vel[2:],
        }
        self._sensor_state = {}
        return self._get_ob()

    def render(self, mode="human"):
        from gym.envs.classic_control import (  # pylint: disable=import-outside-toplevel
            rendering,
        )

        super().render(mode=mode, final=False)

        # arm
        p = self._state["x"][0:2]
        theta = self._state["x"][2]
        q = self._state["q"]
        l, r, t, b = 0, self.LINK_LENGTH, 0.05, -0.05
        p_arm = p + 0.2 * np.array([np.cos(theta), np.sin(theta)])
        tf_arm = rendering.Transform(rotation=theta + q, translation=p_arm)
        link = self._viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.2, 0.8)
        link.add_attr(tf_arm)
        time.sleep(self.dt())

        return self._viewer.render(return_rgb_array=mode == "rgb_array")
