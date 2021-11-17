import numpy as np
import time

from groundRobots.envs.groundRobotEnv import GroundRobotEnv


class GroundRobotArmEnv(GroundRobotEnv):
    LINK_LENGTH = 1.0
    MAX_ARM_VEL = 4 * np.pi
    MAX_ARM_POS = np.pi
    MAX_ARM_ACC = 9 * np.pi

    def __init__(self, render=False, dt=0.01, n_arm=1):
        self._n_arm = n_arm
        self._limUpArmPos = np.ones(self._n_arm) * self.MAX_ARM_POS
        self._limUpArmVel = np.ones(self._n_arm) * self.MAX_ARM_VEL
        self._limUpArmAcc = np.ones(self._n_arm) * self.MAX_ARM_ACC
        super().__init__(render=render, dt=dt)

    def reset(self, pos=None, vel=None):
        """ The velocity is the forward velocity and turning velocity here """
        if not isinstance(pos, np.ndarray) or not pos.size == 3 + self._n_arm:
            pos = np.zeros(3 + self._n_arm)
        if not isinstance(vel, np.ndarray) or not vel.size == 2 + self._n_arm:
            vel = np.zeros(2 + self._n_arm)
        self.state = np.concatenate((pos, vel))
        return self._get_ob()

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        bound_x = self.MAX_POS_BASE + 1.0
        bound_y = self.MAX_POS_BASE + 1.0
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bound_x, bound_x, -bound_y, bound_y)
        self.viewer.draw_line((-bound_x, 0.0), (bound_x, 0.0))
        self.viewer.draw_line((0.0, -bound_y), (0.0, bound_y))

        if s is None:
            return None

        p = [s[0], s[1]]

        theta = s[2]
        tf = rendering.Transform(rotation=theta, translation=p)

        l, r, t, b = (
            -0.5 * self.BASE_LENGTH,
            0.5 * self.BASE_LENGTH,
            0.5 * self.BASE_WIDTH,
            -0.5 * self.BASE_WIDTH,
        )
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        yw = self.BASE_WHEEL_DIST / 2.0
        wheelfl = self.viewer.draw_polygon(
            [(0.2, yw), (0.2, yw + 0.1), (0.4, yw + 0.1), (0.4, yw)]
        )
        wheelfr = self.viewer.draw_polygon(
            [(0.2, -yw), (0.2, -yw - 0.1), (0.4, -yw - 0.1), (0.4, -yw)]
        )
        wheelbl = self.viewer.draw_polygon(
            [(-0.2, yw), (-0.2, yw + 0.1), (-0.4, yw + 0.1), (-0.4, yw)]
        )
        wheelbr = self.viewer.draw_polygon(
            [(-0.2, -yw), (-0.2, -yw - 0.1), (-0.4, -yw - 0.1), (-0.4, -yw)]
        )
        wheelfl.add_attr(tf)
        wheelfr.add_attr(tf)
        wheelbl.add_attr(tf)
        wheelbr.add_attr(tf)
        link.set_color(0, 0.8, 0.8)
        link.add_attr(tf)

        # arm
        q = self.state[3]
        l, r, t, b = 0, self.LINK_LENGTH, 0.05, -0.05
        p_arm = p + 0.2 * np.array([np.cos(theta), np.sin(theta)])
        tf_arm = rendering.Transform(rotation=theta + q, translation=p_arm)
        link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
        link.set_color(0, 0.2, 0.8)
        link.add_attr(tf_arm)
        time.sleep(self.dt())

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
