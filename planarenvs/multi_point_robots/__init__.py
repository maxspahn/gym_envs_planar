from gym.envs.registration import register

register(
    id="multi-point-robots-acc-v0",
    entry_point="planarenvs.multi_point_robots.envs:MultiPointRobotsAccEnv",
)
register(
    id="multi-point-robots-vel-v0",
    entry_point="planarenvs.multi_point_robots.envs:MultiPointRobotsVelEnv",
)
