import numpy as np
from gym import spaces

from mobileRobot.envs.mobileRobotEnv import MobileRobotEnv
from planarCommon.velEnv import VelEnv

class MobileRobotVelEnv(VelEnv, MobileRobotEnv):
    pass
