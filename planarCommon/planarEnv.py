from abc import abstractmethod
import numpy as np
from scipy.integrate import odeint

from gym import core
from gym.utils import seeding

class WrongObservationError(Exception):
    pass


class PlanarEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    def __init__(self, render=False, dt=0.01):
        self.viewer = None
        self.state = None
        self.seed()
        self._dt = dt
        self._t = 0.0
        self._render = render
        self._obsts = []
        self._goals = []

    @abstractmethod
    def setSpaces(self):
        pass

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def addObstacle(self, obst):
        self._obsts.append(obst)

    def addGoal(self, goal):
        self._goals.append(goal)

    def t(self):
        return self._t

    def resetCommon(self):
        self._obsts = []
        self._goals = []
        self._t = 0.0

    def reset(self, pos=None, vel=None):
        self.resetCommon()
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        self.state = {'x': pos, 'xdot': vel}
        return self._get_ob()

    def step(self, a):
        self.action = a
        self.integrate()
        terminal = self._terminal()
        reward = self._reward()
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})


    @abstractmethod
    def _reward(self):
        pass

    def _get_ob(self):
        if not self.observation_space.contains(self.state):
            __import__('pdb').set_trace()
            raise WrongObservationError("The observation does not fit the defined observation space")
        return self.state

    @abstractmethod
    def _terminal(self):
        pass

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def integrate(self):
        self._t += self.dt()
        t = np.arange(0, 2 * self._dt, self._dt)
        x0 = np.concatenate((self.state['x'], self.state['xdot']))
        ynext = odeint(self.continuous_dynamics, x0, t)
        self.state['x'] = ynext[1][0:self._n]
        self.state['xdot'] = ynext[1][self._n:2*self._n]

    @abstractmethod
    def render(self, mode="human"):
        pass

    def renderCommon(self, bounds):
        from gym.envs.classic_control import rendering
        if self.state is None:
            return None
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-bounds[0], bounds[1], -bounds[1], bounds[1])
        for obst in self._obsts:
            obst.renderGym(self.viewer, t=self.t())
        for goal in self._goals:
            goal.renderGym(self.viewer, t=self.t())

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
