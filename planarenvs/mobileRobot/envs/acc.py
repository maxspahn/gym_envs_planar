import numpy as np
from gym import spaces

from planarenvs.mobileRobot.envs.mobileRobotEnv import MobileRobotEnv
from planarenvs.planarCommon.accEnv import AccEnv


class MobileRobotAccEnv(AccEnv, MobileRobotEnv):
    pass

