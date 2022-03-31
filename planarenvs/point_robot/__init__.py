from gym.envs.registration import register

register(
    id="point-robot-acc-v0",
    entry_point="planarenvs.point_robot.envs:PointRobotAccEnv",
)
register(
    id="point-robot-vel-v0",
    entry_point="planarenvs.point_robot.envs:PointRobotVelEnv",
)
