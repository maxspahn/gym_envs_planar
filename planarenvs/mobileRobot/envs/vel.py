import numpy as np
from gym import spaces

from planarenvs.mobileRobot.envs.mobileRobotEnv import MobileRobotEnv
from planarenvs.planarCommon.velEnv import VelEnv

class MobileRobotVelEnv(VelEnv, MobileRobotEnv):
    pass
