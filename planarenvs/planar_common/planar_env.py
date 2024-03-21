from abc import abstractmethod
import time
import pygame
from pygame import gfxdraw
import numpy as np
from scipy.integrate import odeint
import warnings

from gym import core
from gym.utils import seeding
from gym import spaces

from mpscenes.goals.goal_composition import GoalComposition


class WrongObservationError(Exception):
    """Exception when observation lays outside the defined observation space.

    This Exception is initiated when an the observation is not within the
    defined observation space. The purpose of this exception is to give
    the user better information about which specific part of the observation
    caused the problem.
    """

    def __init__(self, msg: str, observation: dict, observationSpace):
        """Constructor for error message.

        Parameters
        ----------

        msg: Default error message
        observation: Observation when mismatch occured
        observationSpace: Observation space of environment
        """
        msg_ext = self.get_wrong_observation(observation, observationSpace)
        super().__init__(msg + msg_ext)

    def get_wrong_observation(self, o: dict, os) -> str:
        """Detecting where the error occured.

        Parameters
        ----------

        o: observation
        os: observation space
        """
        msg_ext = ":\n"
        msg_ext += self.check_dict(o, os)
        return msg_ext

    def check_dict(
        self, o_dict: dict, os_dict, depth: int = 1, tabbing: str = ""
    ) -> str:
        """Checking correctness of dictionary observation.

        This methods searches for the cause for wrong observation.
        It loops over all keys in this dictionary and verifies whether
        observation and observation spaces fit together. If this is not
        the case, the concerned key is checked again. As the observation
        might have nested dictionaries, this function is called
        recursively.

        Parameters
        ----------

        o_dict: observation dictionary
        os_dict: observation space dictionary
        depth: current depth of nesting
        tabbing: tabbing for error message
        """
        msg_ext = ""
        for key in o_dict.keys():
            if not os_dict[key].contains(o_dict[key]):
                if isinstance(o_dict[key], dict):
                    msg_ext += tabbing + key + "\n"
                    msg_ext += self.check_dict(
                        o_dict[key],
                        os_dict[key],
                        depth=depth + 1,
                        tabbing=tabbing + "\t",
                    )
                else:
                    msg_ext += self.check_box(
                        o_dict[key], os_dict[key], key, tabbing
                    )
        return msg_ext

    def check_box(
        self, o_box: np.ndarray, os_box, key: str, tabbing: str
    ) -> str:
        """Checks correctness of box observation.

        This methods detects which value in the observation caused the
        error to be raised. Then it updates the error message msg.

        Parameters
        ----------

        o_box: observation box
        os_box: observation space box
        key: key of observation
        tabbing: current tabbing for error message
        """
        msg_ext = tabbing + "Error in " + key + "\n"
        if isinstance(o_box, float):
            val = o_box
            if val < os_box.low[0]:
                msg_ext += f"{tabbing}\t{key}: {val} < {os_box.low[0]}\n"
            elif val > os_box.high[0]:
                msg_ext += f"{tabbing}\t{key}: {val} > {os_box.high[0]}\n"
            return msg_ext

        for i, val in enumerate(o_box):
            if val < os_box.low[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} < {os_box.low[i]}\n"
            elif val > os_box.high[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} > {os_box.high[i]}\n"
        return msg_ext


class PlanarEnv(core.Env):

    SCREEN_DIM = 30

    def __init__(self, render: bool = False, dt=0.01):
        self._viewer = None
        self._state = {
            "joint_state": {
                "position": None,
                "velocity": None
            }
        }
        self._sensor_state = None
        self.seed()
        self._dt = dt
        self._t = 0.0
        self._render = render
        self._obsts = []
        self._goals = []
        self._sensors = []
        self._outside_limits = False
        self.observation_space = None
        self._n = None

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None


    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @n.deleter
    def n(self):
        del self._n

    @abstractmethod
    def set_spaces(self):
        pass

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obstacle(self, obst):
        self._obsts.append(obst)

    def add_goal(self, goal):
        self._goals.append(goal)

    def add_sensor(self, sensor):
        self._sensors.append(sensor)
        observation_space_dict = dict(self.observation_space.spaces)
        observation_space_dict[sensor.name] = sensor.observation_space()
        self.observation_space = spaces.Dict(observation_space_dict)

    def t(self):
        return self._t

    def reset_common(self):
        self._obsts = []
        self._goals = []
        self._sensors = []
        self._t = 0.0

    def reset(self, pos: np.ndarray = None, vel: np.ndarray = None) -> dict:
        self.reset_common()
        if not isinstance(pos, np.ndarray) or not pos.size == self._n:
            pos = np.zeros(self._n)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        self._state['joint_state']['position'] = pos
        self._state['joint_state']['velocity'] = vel
        self._sensor_state = {}
        return self._get_ob()

    def step(self, action: np.ndarray) -> tuple:
        self._action = action
        self.integrate()
        for sensor in self._sensors:
            self._sensor_state = sensor.sense(
                self._state, self._goals, self._obsts, self.t()
            )
        terminal = self._terminal()
        reward = self._reward()
        if self._render:
            self.render()
        return (self._get_ob(), reward, terminal, {})

    @abstractmethod
    def _reward(self):
        pass

    def _get_ob(self):
        observation = dict(self._state)
        observation.update(self._sensor_state)
        if not self.observation_space['joint_state'].contains(observation['joint_state']):
            err = WrongObservationError(
                "The observation does not fit the defined observation space",
                observation['joint_state'],
                self.observation_space['joint_state'],
            )
            self._outside_limits = True
            warnings.warn(str(err))
        return observation

    def _terminal(self):
        return self._outside_limits

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def integrate(self):
        self._t += self.dt()
        t = np.arange(0, 2 * self._dt, self._dt)
        x0 = np.concatenate((
                self._state["joint_state"]["position"],
                self._state["joint_state"]["velocity"],
            ))
        ynext = odeint(self.continuous_dynamics, x0, t)
        self._state["joint_state"]["position"] = ynext[1][0 : self._n]
        self._state["joint_state"]["velocity"] = ynext[1][self._n: 2 * self._n]


    @abstractmethod
    def render_specific(self, mode="human"):
        pass

    def render_line(self, point_a, point_b, angle=0, color=(0, 0, 0)):
        c, s = np.cos(angle), np.sin(angle)
        tf_matrix = np.array(((c, -s), (s, c)))
        point_a = np.dot(tf_matrix, point_a)
        point_b = np.dot(tf_matrix, point_b)
        point_a[0] *= self.SCREEN_DIM
        point_a[1] *= self.SCREEN_DIM
        point_b[0] *= self.SCREEN_DIM
        point_b[1] *= self.SCREEN_DIM
        point_a[0] += self._offsets[0]
        point_a[1] += self._offsets[1]
        point_b[0] += self._offsets[0]
        point_b[1] += self._offsets[1]
        pygame.draw.line(
            self.surf,
            start_pos=point_a,
            end_pos=point_b,
            color=color,
        )

    def render_polygone(self, coordinates, color=(204, 204, 0)):
        for coordinate in coordinates:
            coordinate[0] = coordinate[0] * self.SCREEN_DIM + self._offsets[0]
            coordinate[1] = coordinate[1] * self.SCREEN_DIM + self._offsets[1]
        gfxdraw.filled_polygon(self.surf, coordinates, color)

    def render_rectangle(self, position, size, color=(204, 204, 0)):
        corners = [
            [position[0] - size[0] / 2, position[1] - size[1] / 2],
            [position[0] + size[0] / 2, position[1] - size[1] / 2],
            [position[0] + size[0] / 2, position[1] + size[1] / 2],
            [position[0] - size[0] / 2, position[1] + size[1] / 2],
        ]
        self.render_polygone(corners, color=color)



    def render_point(self, point_a, color=(204, 204, 0), radius=0.1):
        pos_x = point_a[0] * self.SCREEN_DIM + self._offsets[0]
        pos_y = point_a[1] * self.SCREEN_DIM + self._offsets[1]
        gfxdraw.filled_circle(
            self.surf,
            int(pos_x),
            int(pos_y),
            int(radius * self.SCREEN_DIM),
            color,
        )
    def render_sub_goal(self, sub_goal):
        if sub_goal.dimension() == 2:
            self.render_point(
                sub_goal.position(t=self.t()),
                radius=sub_goal.epsilon(),
                color=(0, 255, 0)
            )
        if sub_goal.dimension() == 1:
            angle = sub_goal.angle()
            if sub_goal.indices()[0] == 0:
                self.render_line(
                    [sub_goal.position(t=self.t())[0], -10],
                    [sub_goal.position(t=self.t())[0], 10],
                    angle=angle,
                    color=(0, 255, 0)
                )
            if sub_goal.indices()[0] == 1:
                self.render_line(
                    [-3, sub_goal.position(t=self.t())[0]],
                    [3, sub_goal.position(t=self.t())[0]],
                    angle=angle,
                    color=(0, 255, 0)
                )

    def render_scene(self):
        for obst in self._obsts:
            if obst.type() == 'box':
                self.render_rectangle(
                    obst.position(t=self.t()),
                    obst.size(),
                    color=(0, 0, 0),
                )
            if obst.type() == 'sphere':
                self.render_point(
                    obst.position(t=self.t()),
                    radius=obst.radius(),
                    color=(0, 0,0)
                )
        for goal in self._goals:
            if isinstance(goal, GoalComposition):
                for sub_goal in goal.sub_goals():
                    self.render_sub_goal(sub_goal)
            else:
                self.render_sub_goal(goal)

    def render(self):
        if self.screen is None:
            scale_x = self._limits["pos"]["high"][0] - self._limits["pos"]["low"][0]
            scale_y = self._limits["pos"]["high"][1] - self._limits["pos"]["low"][1]
            #self._scale = self.SCREEN_DIM / (
            #    self._limits["pos"]["high"][0] - self._limits["pos"]["low"][0]
            #)
            #self._offset = self.SCREEN_DIM / (2 * self._scale)
            self._scales = [self.SCREEN_DIM * 0.5 * scale_x, self.SCREEN_DIM * 0.5 * scale_y]
            self._offsets = [
                (0 - self._limits['pos']['low'][0])/scale_x * self.SCREEN_DIM * scale_x, 
                (0 - self._limits['pos']['low'][1])/scale_y * self.SCREEN_DIM * scale_y, 
            ]
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (2 * self._scales[0], 2 * self._scales[1])
            )
        self.surf = pygame.Surface((2 * self._scales[0], 2 * self._scales[1]))
        self.surf.fill((255, 255, 255))
        self.render_specific()
        self.render_scene()
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        pygame.event.pump()
        pygame.display.flip()
        time.sleep(self.dt())

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
