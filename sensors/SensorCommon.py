from abc import ABC, abstractmethod
import numpy as np
from gym import spaces


def dist2circ(pointPos, circlePos, circleRadius):
    diff = pointPos - circlePos
    dist2center = (diff[0] ** 2 + diff[1] ** 2) ** (1 / 2)
    if dist2center < circleRadius:
        return np.zeros(2)
    diff = diff * (dist2center - circleRadius) / dist2center
    return diff


class Sensor(ABC):

    def __init__(self, nbObservations, limSensor=10):
        self._nbObservations = nbObservations
        self._limSensor = limSensor
        self._observation = []

    @abstractmethod
    def name(self):
        pass

    def getOSpaceSize(self):
        return self._nbObservations, 2

    def getObservationSpace(self):
        return spaces.Box(low=-self._limSensor, high=self._limSensor, shape=self.getOSpaceSize(), dtype=np.float64)

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def sense(self, s, goals, obstacles, t=0):
        pass
