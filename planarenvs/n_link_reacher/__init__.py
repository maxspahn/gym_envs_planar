from gym.envs.registration import register

register(
    id="nLink-reacher-acc-v0",
    entry_point="planarenvs.n_link_reacher.envs:NLinkAccReacherEnv",
)
register(
    id="nLink-reacher-vel-v0",
    entry_point="planarenvs.n_link_reacher.envs:NLinkVelReacherEnv",
)
