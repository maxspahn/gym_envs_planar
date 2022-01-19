from gym.envs.registration import register
register(
    id='mobile-robot-vel-v0',
    entry_point='mobileRobot.envs:MobileRobotVelEnv'
)
register(
    id='mobile-robot-acc-v0',
    entry_point='mobileRobot.envs:MobileRobotAccEnv'
)
