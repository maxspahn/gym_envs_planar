import operator
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


class GoalSensor(object):

    def __init__(self, nbGoals=0, limSensor=10, mode="position"):
        self._mode = mode
        self._nbGoals = nbGoals
        self._limSensor = limSensor
        self._goalObs = []
        if self._mode == "position":
            self._name = "GoalPosition"
        elif self._mode == "distance":
            self._name = "GoalDistance"
        else:
            self._name = "UNDEFINED GOAL"

    def name(self):
        return self._name

    def getOSpaceSize(self):
        return self._nbGoals, 2

    def getObservationSpace(self):
        return spaces.Box(low=-self._limSensor, high=self._limSensor, shape=self.getOSpaceSize(), dtype=np.float64)

    def _reset(self):
        self._goalObs = []

    def sense(self, s, goals, t=0):
        self._reset()
        for idx, goal in enumerate(goals):
            if self._mode == "position":
                currGoalPos = np.clip(goal.position(t=t), -self._limSensor, self._limSensor)
                self._goalObs.append(currGoalPos)
            elif self._mode == "distance":
                currGoalPos = np.clip(goal.position(t=t), -self._limSensor, self._limSensor)
                currGoalDist = Dist2Circ(s['x'][0], s['x'][1], currGoalPos[0], currGoalPos[1], goal.epsilon())
                currGoalDist = np.clip(currGoalDist, -self._limSensor, self._limSensor)
                self._goalObs.append(currGoalDist)
        return self._goalObs


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
        return self._obsts.clip(-self._limSensor, self._limSensor)


class PseudoDistSensor(object):
    ### NOT WORKING YET
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
        # 1) sort list of obstacles, closes obst first
        # 2) take only the nbObs closest obstacles into account
        # 3) store them in the sensor observation

        ## double check if this works
        currPos = s['x']  # only true for point_robot
        obstaclesInfo = []
        for idOb, obstacle in enumerate(obstacles):
            currObstPos = obstacle.position(t=t)
            currObstDist = Dist2Circ(currPos[0], currPos[1], currObstPos[0], currObstPos[1], obstacle.radius())
            currAbsObstDist = abs(currObstDist[0]) + abs(currObstDist[1])
            obstaclesInfo.append({'id': idOb, 'currObstDist': currObstDist, 'currObstPos': currObstPos, 'currAbsObstDist': currAbsObstDist})
        obstaclesInfo.sort(reverse=False, key=operator.itemgetter('currAbsObstDist'))

        self._reset()
        for idx, obst in enumerate(obstaclesInfo):
            if idx >= self._nbObs:
                break
            self._obsts[idx][0] = obst['currObstDist'][0]
            self._obsts[idx][1] = obst['currObstDist'][1]
        return self._obsts

class PseudoGoalObsSensor(object):
    ## NOT WORKING YET!!!
    def __init__(self, obstacles=False, goals=True, nbObs=0, limSensor=10):
        ## todo: implement obstacle and goal flag.
        self._goalSensing = goals
        self._obstSensing = obstacles
        self._nbObs = nbObs
        self._limSensor = limSensor
        self._obsts = np.ones([self._nbObs, 2]) * self._limSensor
        self._goals = None
        self._name = "PseudoSensor"

    def name(self):
        return self._name

    def getOSpaceSize(self):
        return self._nbObs, 2

    def getObservationSpace(self):
        return spaces.Dict({
            'goals': spaces.Box(low=-self._limSensor, high=self._limSensor, shape=(1,2), dtype=np.float64),
            'obstacles': spaces.Box(low=-self._limSensor, high=self._limSensor, shape=self.getOSpaceSize(), dtype=np.float64)
        })
            #spaces.Box(low=-self._limSensor, high=self._limSensor, shape=self.getOSpaceSize(), dtype=np.float64)

    def _reset(self):
        self._obsts[:] = self._limSensor
        self._goals = None

    def sense(self, s=None, obstacles=[], goals=[], t=0):

        if self._goalSensing:
            print('goals')
            for idx, goal in enumerate(goals):
                if idx >= self._nbObs:
                    break
                self._goals[idx][0] = goal.position(t=t)[0]
                self._goals[idx][1] = goal.position(t=t)[1]

        if self._obstSensing:
            self._reset()
            for idx, obst in enumerate(obstacles):
                if idx >= self._nbObs:
                    break
                self._obsts[idx][0] = obst.position(t=t)[0]
                self._obsts[idx][1] = obst.position(t=t)[1]

        observation = {'goals': self._goals, 'obstacles': self._obsts}
        return observation