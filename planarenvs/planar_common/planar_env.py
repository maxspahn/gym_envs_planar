from abc import abstractmethod
import numpy as np
from scipy.integrate import odeint
import warnings

from gym import core
from gym.utils import seeding
from gym import spaces


class WrongObservationError(Exception):
    """Exception when observation lays outside the defined observation space.

    This Exception is initiated when an the observation is not within the
    defined observation space. The purpose of this exception is to give
    the user better information about which specific part of the observation
    caused the problem.
    """

    def __init__(self, msg: str, observation: dict, observationSpace):
        """Constructor for error message.

        Parameters
        ----------

        msg: Default error message
        observation: Observation when mismatch occured
        observationSpace: Observation space of environment
        """
        msg_ext = self.get_wrong_observation(observation, observationSpace)
        super().__init__(msg + msg_ext)

    def get_wrong_observation(self, o: dict, os) -> str:
        """Detecting where the error occured.

        Parameters
        ----------

        o: observation
        os: observation space
        """
        msg_ext = ":\n"
        msg_ext += self.check_dict(o, os)
        return msg_ext

    def check_dict(
        self, o_dict: dict, os_dict, depth: int = 1, tabbing: str = ""
    ) -> str:
        """Checking correctness of dictionary observation.

        This methods searches for the cause for wrong observation.
        It loops over all keys in this dictionary and verifies whether
        observation and observation spaces fit together. If this is not
        the case, the concerned key is checked again. As the observation
        might have nested dictionaries, this function is called
        recursively.

        Parameters
        ----------

        o_dict: observation dictionary
        os_dict: observation space dictionary
        depth: current depth of nesting
        tabbing: tabbing for error message
        """
        msg_ext = ""
        for key in o_dict.keys():
            if not os_dict[key].contains(o_dict[key]):
                if isinstance(o_dict[key], dict):
                    msg_ext += tabbing + key + "\n"
                    msg_ext += self.check_dict(
                        o_dict[key],
                        os_dict[key],
                        depth=depth + 1,
                        tabbing=tabbing + "\t",
                    )
                else:
                    msg_ext += self.check_box(
                        o_dict[key], os_dict[key], key, tabbing
                    )
        return msg_ext

    def check_box(
        self, o_box: np.ndarray, os_box, key: str, tabbing: str
    ) -> str:
        """Checks correctness of box observation.

        This methods detects which value in the observation caused the
        error to be raised. Then it updates the error message msg.

        Parameters
        ----------

        o_box: observation box
        os_box: observation space box
        key: key of observation
        tabbing: current tabbing for error message
        """
        msg_ext = tabbing + "Error in " + key + "\n"
        if isinstance(o_box, float):
            val = o_box
            if val < os_box.low[0]:
                msg_ext += f"{tabbing}\t{key}: {val} < {os_box.low[0]}\n"
            elif val > os_box.high[0]:
                msg_ext += f"{tabbing}\t{key}: {val} > {os_box.high[0]}\n"
            return msg_ext

        for i, val in enumerate(o_box):
            if val < os_box.low[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} < {os_box.low[i]}\n"
            elif val > os_box.high[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} > {os_box.high[i]}\n"
        return msg_ext


class PlanarEnv(core.Env):
    def __init__(self, render: bool = False, dt=0.01):
        self._viewer = None
        self._state = {
            'joint_state': {
                'position': None,
                'velocity': None
            }
        }
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
        self._state['joint_state']['position'] = pos
        self._state['joint_state']['velocity'] = vel
        self._sensor_state = {}
        return self._get_ob()

    def step(self, action: np.ndarray) -> tuple:
        self._action = action
        self.integrate()
        for sensor in self._sensors:
            self._sensor_state = sensor.sense(
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
        if not self.observation_space['joint_state'].contains(observation['joint_state']):
            err = WrongObservationError(
                "The observation does not fit the defined observation space",
                observation['joint_state'],
                self.observation_space['joint_state'],
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
        x0 = np.concatenate((
                self._state['joint_state']['position'], 
                self._state['joint_state']['velocity'], 
            ))
        ynext = odeint(self.continuous_dynamics, x0, t)
        self._state["joint_state"]["position"] = ynext[1][0 : self._n]
        self._state["joint_state"]["velocity"] = ynext[1][self._n: 2 * self._n]


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
            obst.render_gym(self._viewer, rendering, t=self.t())
        for goal in self._goals:
            goal.render_gym(self._viewer, rendering, t=self.t())

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
