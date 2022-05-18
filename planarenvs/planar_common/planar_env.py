from abc import abstractmethod
import numpy as np
from scipy.integrate import odeint
import warnings

from gym import core
from gym.utils import seeding
from gym import spaces


class WrongObservationError(Exception):
    def __init__(self, msg, observation, observation_space):
        msg_ext = self.get_wrong_observation(observation, observation_space)
        super().__init__(msg + msg_ext)

    def get_wrong_observation(
        self, observation: dict, observation_space: spaces.Dict
    ) -> str:
        msg_ext = ": "
        for key in observation.keys():
            if not observation_space[key].contains(observation[key]):
                msg_ext += "Error in " + key
                for i, val in enumerate(observation[key]):
                    if val < observation_space[key].low[i]:
                        msg_ext += (
                            f"[{i}]: {val} < {observation_space[key].low[i]}"
                        )
                    elif val > observation_space[key].high[i]:
                        msg_ext += (
                            f"[{i}]: {val} > {observation_space[key].high[i]}"
                        )
        return msg_ext


class PlanarEnv(core.Env):
    def __init__(self, render: bool = False, dt=0.01):
        self._viewer = None
        self._state = {'x': None, 'xdot': None}
        self._sensor_state = None
        self.seed()
        self._dt = dt
        self._t = 0.0
        self._render = render
        self._obsts = []
        self._goals = []
        self._sensors = []
        self.observation_space = None
        self._n = None

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @n.deleter
    def n(self):
        del self._n

    @abstractmethod
    def set_spaces(self):
        pass

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obstacle(self, obst):
        self._obsts.append(obst)

    def add_goal(self, goal):
        self._goals.append(goal)

    def add_sensor(self, sensor):
        self._sensors.append(sensor)
        observation_space_dict = dict(self.observation_space.spaces)
        print(sensor.name)
        observation_space_dict[sensor.name] = sensor.observation_space()
        self.observation_space = spaces.Dict(observation_space_dict)

    def t(self):
        return self._t

    def reset_common(self):
        self._obsts = []
        self._goals = []
        self._sensors = []
        self._t = 0.0

    def reset(self, pos: np.ndarray = None, vel: np.ndarray = None) -> dict:
        self.reset_common()
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        self._state = {"x": pos, "xdot": vel}
        self._sensor_state = {}
        return self._get_ob()

    def step(self, action: np.ndarray) -> tuple:
        self._action = action
        self.integrate()
        for sensor in self._sensors:
            self._sensor_state[sensor.name] = sensor.sense(
                self._state, self._goals, self._obsts, self.t()
            )
        terminal = self._terminal()
        reward = self._reward()
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    @abstractmethod
    def _reward(self):
        pass

    def _get_ob(self):
        observation = dict(self._state)
        observation.update(self._sensor_state)
        if not self.observation_space.contains(observation):
            err = WrongObservationError(
                "The observation does not fit the defined observation space",
                observation,
                self.observation_space,
            )
            warnings.warn(str(err))
        return observation

    @abstractmethod
    def _terminal(self):
        pass

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def integrate(self):
        self._t += self.dt()
        t = np.arange(0, 2 * self._dt, self._dt)
        x0 = np.concatenate((self._state["x"], self._state["xdot"]))
        ynext = odeint(self.continuous_dynamics, x0, t)
        self._state["x"] = ynext[1][0 : self._n]
        self._state["xdot"] = ynext[1][self._n : 2 * self._n]

    @abstractmethod
    def render(self, mode="human"):
        pass

    def render_common(self, bounds):
        from gym.envs.classic_control import (
            rendering,
        )  # pylint: disable=import-outside-toplevel

        if self._state is None:
            return None
        if self._viewer is None:
            if isinstance(bounds, list):
                self._viewer = rendering.Viewer(500, 500)
                self._viewer.set_bounds(
                    -bounds[0], bounds[1], -bounds[1], bounds[1]
                )
            elif isinstance(bounds, dict):
                ratio = (bounds["pos"]["high"][0] - bounds["pos"]["low"][0]) / (
                    bounds["pos"]["high"][1] - bounds["pos"]["low"][1]
                )
                if ratio > 1:
                    window_size = (1000, int(1000 / ratio))
                else:
                    window_size = (int(ratio * 1000), 1000)
                self._viewer = rendering.Viewer(window_size[0], window_size[1])
                self._viewer.set_bounds(
                    bounds["pos"]["low"][0],
                    bounds["pos"]["high"][0],
                    bounds["pos"]["low"][1],
                    bounds["pos"]["high"][1],
                )
        for obst in self._obsts:
            obst.renderGym(self._viewer, rendering, t=self.t())
        for goal in self._goals:
            goal.renderGym(self._viewer, rendering, t=self.t())

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
