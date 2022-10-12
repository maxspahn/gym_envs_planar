from planarenvs.sensors.sensor_common import Sensor
import numpy as np
from gym import spaces


class FullSensor(Sensor):
    def __init__(self, goal_mask: list, obstacle_mask: list, variance: int= 0.1):
        self._obstacle_mask = obstacle_mask
        self._goal_mask = goal_mask
        self._name = "FullSensor"
        self._noise_variance = variance

    def _reset(self):
        pass

    def observation_size(self) -> tuple:
        return 1, 2

    def observation_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.observation_size(),
            dtype=np.float64,
        )

    def sense(self, state, goals, obstacles: list, t=0):
        sensor_observation = {"goals": [], "obstacles": []}
        for goal in goals:
            exact_observations = list(goal.evaluate_components(self._goal_mask, t).values())
            noisy_observations = []
            for exact_observation in exact_observations:
                if isinstance(exact_observation, np.ndarray):
                    noisy_observation = np.random.normal(exact_observation, self._noise_variance)
                else:
                    noisy_observation = exact_observation
                noisy_observations.append(noisy_observation)
            sensor_observation["goals"].append(noisy_observations)
        for obstacle in obstacles:
            exact_observations = list(obstacle.evaluate_components(self._obstacle_mask, t).values())
            noisy_observations = []
            for exact_observation in exact_observations:
                if isinstance(exact_observation, np.ndarray):
                    noisy_observation = np.random.normal(exact_observation, self._noise_variance)
                else:
                    noisy_observation = exact_observation
                noisy_observations.append(noisy_observation)
            sensor_observation["obstacles"].append(noisy_observations)
        return sensor_observation
