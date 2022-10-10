from planarenvs.sensors.sensor_common import Sensor
from planarenvs.sensors.sensor_common import dist2circ
import numpy as np
import operator


class ObstacleSensor(Sensor):
    def __init__(self, nb_obstacles=0, lim_sensor=10, mode="position"):
        super().__init__(nb_observations=nb_obstacles, lim_sensor=lim_sensor)
        self._observation = (
            np.ones([self._nb_observations, 2]) * self._lim_sensor
        )
        self._mode = mode
        self.name = mode

    @Sensor.name.setter
    def name(self, mode):
        if mode == "position":
            self._name = "ObstaclePosition"
        elif mode == "distance":
            self._name = "ObstacleDistance"

    def _reset(self):
        self._observation[:] = self._lim_sensor

    def sense(
        self, state: dict, goals, obstacles, t=0
    ) -> dict:  # pylint: disable=unused-argument
        self._reset()
        if self._mode == "position":
            for idx, obst in enumerate(obstacles):
                if idx >= self._nb_observations:
                    break
                self._observation[idx] = obst.position(t=t)
        elif self._mode == "distance":
            self._reset()
            current_position = state["joint_state"]["position"]  # only true for point_robot
            obstacles_info = []
            for id_obstacle, obstacle in enumerate(obstacles):
                obstacle_position = obstacle.position(t=t)
                obstacle_distance = dist2circ(
                    current_position,
                    obstacle_position,
                    obstacle.radius(),
                )
                absolute_distance = abs(obstacle_distance[0]) + abs(
                    obstacle_distance[1]
                )
                obstacles_info.append(
                    {
                        "id": id_obstacle,
                        "distance": obstacle_distance,
                        "position": obstacle_position,
                        "absolute_distance": absolute_distance,
                    }
                )
            obstacles_info.sort(
                reverse=False, key=operator.itemgetter("absolute_distance")
            )
            for idx, obst in enumerate(obstacles_info):
                if idx >= self._nb_observations:
                    break
                self._observation[idx] = obst["distance"]

        return self._observation.clip(-self._lim_sensor, self._lim_sensor)
