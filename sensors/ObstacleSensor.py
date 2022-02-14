from sensors.AbstractSensor import Sensor
from sensors.AbstractSensor import Dist2Circ
import numpy as np
import operator


class ObstacleSensor(Sensor):

    def __init__(self, obstacles=False, nbObstacles=0, SensorRange=10, mode="position"):
        super().__init__(nbObservations=nbObstacles, SensorRange=SensorRange)
        ## todo: implement obstacle and goal flag.
        self._Observation = np.ones([self._nbObservations, 2]) * self._limSensor
        self._mode = mode
        self._setSensorName()

    def _setSensorName(self):
        if self._mode == "position":
            self._name = "ObstaclePseudoSensorPosition"
        elif self._mode == "distance":
            self._name = "ObstaclePseudoSensorDistance"

    def name(self):
        return self._name

    def _reset(self):
        self._Observation[:] = self._limSensor

    def sense(self, s, goals, obstacles, t=0):
        self._reset()
        if self._mode == "position":
            for idx, obst in enumerate(obstacles):
                if idx >= self._nbObservations:
                    break
                self._Observation[idx][0] = obst.position(t=t)[0]
                self._Observation[idx][1] = obst.position(t=t)[1]
        elif self._mode == "distance":
            self._reset()
            currPos = s['x']  # only true for point_robot
            obstaclesInfo = []
            for idOb, obstacle in enumerate(obstacles):
                currObstPos = obstacle.position(t=t)
                currObstDist = Dist2Circ(currPos[0], currPos[1], currObstPos[0], currObstPos[1], obstacle.radius())
                currAbsObstDist = abs(currObstDist[0]) + abs(currObstDist[1])
                obstaclesInfo.append({'id': idOb, 'currObstDist': currObstDist, 'currObstPos': currObstPos,
                                      'currAbsObstDist': currAbsObstDist})
            obstaclesInfo.sort(reverse=False, key=operator.itemgetter('currAbsObstDist'))
            for idx, obst in enumerate(obstaclesInfo):
                if idx >= self._nbObservations:
                    break
                self._Observation[idx][0] = obst['currObstDist'][0]
                self._Observation[idx][1] = obst['currObstDist'][1]

        return self._Observation.clip(-self._limSensor, self._limSensor)
