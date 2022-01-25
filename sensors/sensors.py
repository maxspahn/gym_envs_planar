import numpy as np
from gym import spaces


class PseudoSensor(object):

    def __init__(self, obstacles=False, nbObs=0, limSensor=10):
        ## todo: implement obstacle and goal flag.
        self._nbObs = nbObs
        self._limSensor = limSensor
        self._obsts = np.ones([self._nbObs, 2]) * self._limSensor
        self._name = "PseudoSensor"

    def name(self):
        return self._name

    def getOSpaceSize(self):
        return self._nbObs, 2

    def getObservationSpace(self):
        return spaces.Box(low=-self._limSensor, high=self._limSensor, shape=self.getOSpaceSize(), dtype=np.float64)

    def _reset(self):
        self._obsts[:] = self._limSensor

    def sense(self, s, obstacles, t=0):
        self._reset()
        for idx, obst in enumerate(obstacles):
            if idx >= self._nbObs:
                break
            self._obsts[idx][0] = obst.position(t=t)[0]
            self._obsts[idx][1] = obst.position(t=t)[1]
        return self._obsts


class PseudoDistSensor(object):

    def __init__(self, obstacles=False, nbObs=0, limSensor=10):
        ## todo: implement obstacle and goal flag.
        self._nbObs = nbObs
        self._limSensor = limSensor
        self._obsts = np.ones([self._nbObs, 2]) * self._limSensor
        self._name = "PseudoDistSensor"

    def name(self):
        return self._name

    def getOSpaceSize(self):
        return self._nbObs, 2

    def _getObstDist(obst):
        ## todo: implement sorting parameter
        return 0

    def getObservationSpace(self):
        return spaces.Box(low=-self._limSensor, high=self._limSensor, shape=self.getOSpaceSize(), dtype=np.float64)

    def _reset(self):
        self._obsts[:] = self._limSensor

    def sense(self, s, obstacles, t=0):
        ## todo: check if boundaries are closer than obstacles
        # 0) compute distance to each obstacle & check if boundaries are closer than obstacles
        curr_pos = s['x']  # only true for point_robot
        # 1) sort list of obstacles, closes obst first
        # obstacles = obstacles.sort(reverse=False, key=getObstDist)
        # 2) take only the nbObs closest obstacles into account
        # 3) store them in the sensor observation
        self._reset()
        for idx, obst in enumerate(obstacles):
            if idx >= self._nbObs:
                break
            self._obsts[id][0] = obst.position(t=t)[0]
            self._obsts[id][1] = obst.position(t=t)[1]
        return self._obsts


