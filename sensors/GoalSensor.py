from sensors.SensorCommon import Sensor
from sensors.SensorCommon import Dist2Circ
import numpy as np
import operator


class GoalSensor(Sensor):

    def __init__(self, nbGoals=0, SensorRange=10, mode="position"):
        super().__init__(nbObservations=nbGoals, SensorRange=SensorRange)
        self._Observation = np.ones([self._nbObservations, 2]) * self._limSensor
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
        self._Observation[:] = 0

    def sense(self, s, goals, obstacles, t=0):
        self._reset()
        if self._mode == "position":
            for idx, goal in enumerate(goals):
                if idx >= self._nbObservations:
                    break
                self._Observation[idx][0] = goal.position(t=t)[0]
                self._Observation[idx][1] = goal.position(t=t)[1]

        elif self._mode == "distance":
            for idx, goal in enumerate(goals):
                if idx >= self._nbObservations:
                    break
                currGoalPos = goal.position(t=t)
                currGoalDist = Dist2Circ(s['x'][0], s['x'][1], currGoalPos[0], currGoalPos[1], goal.epsilon())
                self._Observation[idx][0] = currGoalDist[0]
                self._Observation[idx][1] = currGoalDist[1]

        return self._Observation.clip(-self._limSensor, self._limSensor)


class OldGoalSensor(Sensor):

    def __init__(self, nbGoals=0, SensorRange=10, mode="position"):
        super().__init__(nbObservations=nbGoals, SensorRange=SensorRange)
        self._mode = mode
        self._setSensorName()

    def _setSensorName(self):
        if self._mode == "position":
            self._name = "GoalPosition"
        elif self._mode == "distance":
            self._name = "GoalDistance"

    def name(self):
        return self._name

    def sense(self, s, goals, obstacles, t=0):
        self._reset()
        for idx, goal in enumerate(goals):
            if self._mode == "position":
                currGoalPos = np.clip(goal.position(t=t), -self._limSensor, self._limSensor)
                self._Observation.append(currGoalPos)
            elif self._mode == "distance":
                currGoalPos = np.clip(goal.position(t=t), -self._limSensor, self._limSensor)
                currGoalDist = Dist2Circ(s['x'][0], s['x'][1], currGoalPos[0], currGoalPos[1], goal.epsilon())
                currGoalDist = np.clip(currGoalDist, -self._limSensor, self._limSensor)
                self._Observation.append(currGoalDist)
        return self._Observation