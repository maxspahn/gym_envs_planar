# pylint: disable-all
from planarenvs.n_link_reacher.resources.createDynamics import createDynamics


class AccController(object):

    """Simple forward controller acceleration to torque"""

    def __init__(self, n, l, m, g, k):
        _, _, self.tau_fun = createDynamics(n)
        self._n = n
        self._l = l
        self._m = m
        self._g = g
        self._k = k

    def control(self, q, qdot, qddot):
        tau = self.tau_fun(q, qdot, self._l, self._m, self._g, self._k, qddot)
        return tau
