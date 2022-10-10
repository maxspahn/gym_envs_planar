from planarenvs.sensors.sensor_common import Sensor
from planarenvs.sensors.sensor_common import dist2circ
import numpy as np


class GoalSensor(Sensor):
    def __init__(self, nb_goals=0, lim_sensor=10, mode="position"):
        super().__init__(nb_observations=nb_goals, lim_sensor=lim_sensor)
        self._observation = (
            np.ones([self._nb_observations, 2]) * self._lim_sensor
        )
        self._mode = mode
        self.name = mode

    @Sensor.name.setter
    def name(self, mode):
        if mode == "position":
            self._name = "GoalPosition"
        elif mode == "distance":
            self._name = "GoalDistance"

    def _reset(self):
        self._observation[:] = 0

    def sense(self, state, goals, obstacles, t=0):
        self._reset()
        if self._mode == "position":
            for idx, goal in enumerate(goals):
                if idx >= self._nb_observations:
                    break
                self._observation[idx] = goal.position(t=t)

        elif self._mode == "distance":
            for idx, goal in enumerate(goals):
                if idx >= self._nb_observations:
                    break
                goal_position = goal.position(t=t)
                goal_distance = dist2circ(
                    state["joint_state"]["position"], goal_position, goal.epsilon()
                )
                self._observation[idx] = goal_distance

        return self._observation.clip(-self._lim_sensor, self._lim_sensor)
