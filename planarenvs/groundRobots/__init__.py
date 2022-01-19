from gym.envs.registration import register
register(
    id='ground-robot-vel-v0',
    entry_point='planarenvs.groundRobots.envs:GroundRobotVelEnv'
)
register(
    id='ground-robot-acc-v0',
    entry_point='planarenvs.groundRobots.envs:GroundRobotAccEnv'
)
register(
    id='ground-robot-arm-acc-v0',
    entry_point='planarenvs.groundRobots.envs:GroundRobotArmAccEnv'
)
register(
    id='ground-robot-arm-vel-v0',
    entry_point='planarenvs.groundRobots.envs:GroundRobotArmVelEnv'
)
