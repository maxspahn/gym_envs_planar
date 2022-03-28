from gym.envs.registration import register

register(
    id="mobile-base-vel-v0",
    entry_point="planarenvs.mobile_base.envs:MobileBaseVelEnv",
)
register(
    id="mobile-base-acc-v0",
    entry_point="planarenvs.mobile_base.envs:MobileBaseAccEnv",
)
