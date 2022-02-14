from abc import ABC, abstractmethod
import numpy as np
from gym import spaces


def Dist2Circ(x1, y1, x2,  y2, r):
    dx = x2 - x1
    dy = y2 - y1
    Dist2Center = (dx ** 2 + dy ** 2) ** (1 / 2)
    if Dist2Center < r:
        return np.zeros(2)
    dx = dx * (Dist2Center - r) / Dist2Center
    dy = dy * (Dist2Center - r) / Dist2Center
    return dx, dy


class Sensor(ABC):

    def __init__(self, nbObservations, SensorRange=10):
        self._nbObservations = nbObservations
        self._limSensor = SensorRange
        self._Observation = []

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
