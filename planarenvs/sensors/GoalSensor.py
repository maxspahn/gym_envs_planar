from planarenvs.sensors.SensorCommon import Sensor
from planarenvs.sensors.SensorCommon import dist2circ
import numpy as np
import operator


class GoalSensor(Sensor):

    def __init__(self, nbGoals=0, limSensor=10, mode="position"):
        super().__init__(nbObservations=nbGoals, limSensor=limSensor)
        self._observation = np.ones([self._nbObservations, 2]) * self._limSensor
        self._mode = mode
        self._setSensorName()

    def _setSensorName(self):
        if self._mode == "position":
            self._name = "GoalPosition"
        elif self._mode == "distance":
            self._name = "GoalDistance"

    def name(self):
        return self._name

    def _reset(self):
        self._observation[:] = 0

    def sense(self, state, goals, obstacles, t=0):
        self._reset()
        if self._mode == "position":
            for idx, goal in enumerate(goals):
                if idx >= self._nbObservations:
                    break
                self._observation[idx] = goal.position(t=t)

        elif self._mode == "distance":
            for idx, goal in enumerate(goals):
                if idx >= self._nbObservations:
                    break
                currGoalPos = goal.position(t=t)
                currGoalDist = dist2circ(state['x'], currGoalPos, goal.epsilon())
                self._observation[idx] = currGoalDist

        return self._observation.clip(-self._limSensor, self._limSensor)
