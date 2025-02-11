# pylint: disable=import-outside-toplevel
import gymnasium as gym
from planarenvs.point_robot.envs.vel import PointRobotVelEnv
import numpy as np

# This example showcases the psedo-sensor and requires goals and obstacles.
# As a result, it requires the optional package motion_planning_scenes.

# Consider installing it with `poetry install -E scenes`.


def time_variant_action(t):
    return np.array([np.cos(t), np.sin(t)])


def run_point_robot(
    n_steps: int = 1000,
    render: bool = False,
    goal: bool = False,
    obstacles: bool = False,
):
    """
    Minimal example for point robot in the plane.

    The point robot has two prismatic joints, one for `x` and one for `y`.
    The observation is thus:
        x: [`x`, `y`]
        xdot: [`xdot`, `ydot`]

    In this example, we make use of the pseudo-sensor for detecting obstacles
    and the goal position. The pseudo-sensor adds some observations to the
    observation returned by the `step`-function.
        GoalPosition: position of the goal
        GoalDistance: delta_x and delta_y to the goal
        ObstaclePosition: positions of the obstacles
        ObstacleDistance: delta_x and delta_y to the obstacles
    """
    env:PointRobotVelEnv = gym.make("point-robot-vel-v0", render=render, dt=0.01).unwrapped
    init_pos = np.array([0.5, 0.0])
    init_vel = np.array([0.0, 0.0])
    options = {'pos': init_pos, 'vel': init_vel}
    ob = env.reset(options=options)
    env.reset_limits(
        pos={"high": np.array([2.0, 3.0]), "low": np.array([-2.0, -3.0])}
    )
    sensors = False
    if sensors:
        from planarenvs.sensors.goal_sensor import (
            GoalSensor,
        )
        from planarenvs.sensors.obstacle_sensor import (
            ObstacleSensor,
        )

        obst_sensor_pos = ObstacleSensor(nb_obstacles=2, mode="position")
        env.add_sensor(obst_sensor_pos)
        obst_sensor_dist = ObstacleSensor(nb_obstacles=2, mode="distance")
        env.add_sensor(obst_sensor_dist)
        goal_dist_observer = GoalSensor(nb_goals=1, mode="distance")
        env.add_sensor(goal_dist_observer)
        goal_pos_observer = GoalSensor(nb_goals=1, mode="position")
        env.add_sensor(goal_pos_observer)

        from planarenvs.sensors.full_sensor import FullSensor
        full_sensor = FullSensor(
            ['position', 'velocity', 'is_primary_goal'],
            ['position', 'velocity', 'radius', 'type'],
            variance=0.0
        )
        env.add_sensor(full_sensor)

    if obstacles:
        from planarenvs.scenes.obstacles import (
            sphereObst1,
            sphereObst2,
            dynamicSphereObst1,
            dynamicSphereObst2,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(dynamicSphereObst1)
        env.add_obstacle(dynamicSphereObst2)
    if goal:
        from planarenvs.scenes.goal import (
            splineGoal,
            lineGoal,
            staticGoal,
        )

        env.add_goal(splineGoal)
        env.add_goal(staticGoal)
        env.add_goal(lineGoal)

    print("Starting episode")
    observation_history = []
    for i in range(n_steps):
        action = time_variant_action(env.t())
        ob, _, _, _ = env.step(action)
        observation_history.append(ob)
        if i % 100 == 0:
            print(f"ob : {ob}")
    return observation_history


if __name__ == "__main__":
    obstacles = True
    goal = True
    run_point_robot(render=True, obstacles=obstacles, goal=goal)
