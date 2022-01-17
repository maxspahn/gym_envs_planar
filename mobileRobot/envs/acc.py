import numpy as np
from gym import spaces

from mobileRobot.envs.mobileRobotEnv import MobileRobotEnv
from planarCommon.accEnv import AccEnv


class MobileRobotAccEnv(AccEnv, MobileRobotEnv):
    pass

