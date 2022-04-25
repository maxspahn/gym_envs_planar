from abc import ABC, abstractmethod
import numpy as np
from gym import spaces


def vector_circle_to_point(
    point_position: np.ndarray, circle_position: np.ndarray, circle_radius: float
) -> np.ndarray:
    distance_vector = point_position - circle_position
    return distance_vector

def dist2circ(point_position: np.ndarray, circle_position: np.ndarray, circle_radius: float) -> np.ndarray:
    """
    Computes normalized distance vector between point and circle.

    Returns
    ---------
    np.ndarray:
        (x_point - x_center) * |x_point - x_center|_2 - r_circle) / |x_point - x_center|_2
    """
    distance_vector = vector_circle_to_point(point_position, circle_position, circle_radius)
    distance = np.linalg.norm(distance_vector)
    if distance < circle_radius:
        return np.zeros(2)
    distance_vector_normalized = distance_vector * (distance - circle_radius) / distance
    return distance_vector_normalized



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
    def sense(self, state, goals, obstacles, t=0):
        pass
