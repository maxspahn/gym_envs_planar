from planarenvs.n_link_reacher.envs.n_link_reacher_env import NLinkReacherEnv
from planarenvs.planar_common.acc_env import AccEnv


class NLinkAccReacherEnv(AccEnv, NLinkReacherEnv):
    pass
