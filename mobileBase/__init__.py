from gym.envs.registration import register
register(
    id='mobile-base-vel-v0',
    entry_point='mobileBase.envs:MobileBaseVelEnv'
)
register(
    id='mobile-base-acc-v0',
    entry_point='mobileBase.envs:MobileBaseAccEnv'
)
