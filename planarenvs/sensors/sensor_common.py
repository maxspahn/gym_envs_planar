from abc import ABC, abstractmethod
import numpy as np
from gym import spaces


def dist2circ(
    point_pos: np.ndarray, circle_pos: np.ndarray, circle_radius: float
) -> np.ndarray:
    diff = point_pos - circle_pos
    dist_to_center = np.linalg.norm(diff)
    if dist_to_center < circle_radius:
        return np.zeros(2)
    diff = diff * (dist_to_center - circle_radius) / dist_to_center
    return diff


class Sensor(ABC):
    def __init__(self, nb_observations, lim_sensor=10):
        self._nb_observations = nb_observations
        self._lim_sensor = lim_sensor
        self._observation = []
        self._name = "Sensor"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @name.deleter
    def name(self):
        del self._name

    def observation_size(self) -> tuple:
        return self._nb_observations, 2

    def observation_space(self):
        return spaces.Box(
            low=-self._lim_sensor,
            high=self._lim_sensor,
            shape=self.observation_size(),
            dtype=np.float64,
        )

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def sense(self, s, goals, obstacles, t=0):
        pass
