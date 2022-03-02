from sensors.SensorCommon import Sensor
from sensors.SensorCommon import dist2circ
import numpy as np
import operator


class ObstacleSensor(Sensor):

    def __init__(self, obstacles=False, nbObstacles=0, limSensor=10, mode="position"):
        super().__init__(nbObservations=nbObstacles, limSensor=limSensor)
        ## todo: implement obstacle and goal flag.
        self._observation = np.ones([self._nbObservations, 2]) * self._limSensor
        self._mode = mode
        self._setSensorName()

    def _setSensorName(self):
        if self._mode == "position":
            self._name = "ObstaclePosition"
        elif self._mode == "distance":
            self._name = "ObstacleDistance"

    def name(self):
        return self._name

    def _reset(self):
        self._observation[:] = self._limSensor

    def sense(self, state, goals, obstacles, t=0):
        self._reset()
        if self._mode == "position":
            for idx, obst in enumerate(obstacles):
                if idx >= self._nbObservations:
                    break
                self._observation[idx] = obst.position(t=t)
        elif self._mode == "distance":
            self._reset()
            currPos = state['x']  # only true for point_robot
            obstaclesInfo = []
            for idOb, obstacle in enumerate(obstacles):
                currObstPos = obstacle.position(t=t)
                currObstDist = dist2circ(currPos, currObstPos, obstacle.radius())
                currAbsObstDist = abs(currObstDist[0]) + abs(currObstDist[1])
                obstaclesInfo.append({'id': idOb, 'currObstDist': currObstDist, 'currObstPos': currObstPos,
                                      'currAbsObstDist': currAbsObstDist})
            obstaclesInfo.sort(reverse=False, key=operator.itemgetter('currAbsObstDist'))
            for idx, obst in enumerate(obstaclesInfo):
                if idx >= self._nbObservations:
                    break
                self._observation[idx] = obst['currObstDist']

        return self._observation.clip(-self._limSensor, self._limSensor)
