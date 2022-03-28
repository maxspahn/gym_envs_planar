import gym
import planarenvs.point_robot  # pylint: disable=unused-import
import numpy as np


obstacles = True
goal = True
sensors = True


def main():
    env = gym.make("point-robot-vel-v0", render=True, dt=0.01)
    default_action = np.array([-0.2, 0.10])
    default_action = lambda t: np.array([np.cos(1.0 * t), np.sin(1.0 * t)])
    init_pos = np.array([0.0, -1.0])
    init_vel = np.array([-1.0, 0.0])
    n_steps = 1000
    ob = env.reset(pos=init_pos, vel=init_vel)
    env.reset_limits(
        pos={"high": np.array([1.0, 3.0]), "low": np.array([-1.0, -3.0])}
    )

    if sensors:
        from planarenvs.sensors.goal_sensor import (
            GoalSensor,
        )  # pylint: disable=import-outside-toplevel
        from planarenvs.sensors.obstacle_sensor import (
            ObstacleSensor,
        )  # pylint: disable=import-outside-toplevel

        obst_sensor_pos = ObstacleSensor(nb_obstacles=2, mode="position")
        env.add_sensor(obst_sensor_pos)
        obst_sensor_dist = ObstacleSensor(nb_obstacles=2, mode="distance")
        env.add_sensor(obst_sensor_dist)
        goal_dist_observer = GoalSensor(nb_goals=1, mode="distance")
        env.add_sensor(goal_dist_observer)
        goal_pos_observer = GoalSensor(nb_goals=1, mode="position")
        env.add_sensor(goal_pos_observer)

    if obstacles:
        from examples.obstacles import (  # pylint: disable=import-outside-toplevel
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
        from examples.goal import (
            splineGoal,
        )  # pylint: disable=import-outside-toplevel

        env.add_goal(splineGoal)

    print("Starting episode")
    for _ in range(n_steps):
        action = default_action(env.t())
        ob, _, _, _ = env.step(action)
        print(f"ob : {ob}")


if __name__ == "__main__":
    main()
