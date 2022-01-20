from gym.envs.registration import register
register(
    id='mobile-base-vel-v0',
    entry_point='planarenvs.mobileBase.envs:MobileBaseVelEnv'
)
register(
    id='mobile-base-acc-v0',
    entry_point='planarenvs.mobileBase.envs:MobileBaseAccEnv'
)
