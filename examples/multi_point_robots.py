# pylint: disable=import-outside-toplevel
import gym
import planarenvs.multi_point_robots
import numpy as np

# This example showcases the psedo-sensor and requires goals and obstacles.
# As a result, it requires the optional package motion_planning_scenes.

# Consider installing it with `poetry install -E scenes`.


def time_variant_action(t, n):
    #return np.array([np.cos(t), np.sin(t), 0.0, -1.0])
    return np.ones(n)


def run_multi_point_robots(
    n_steps: int = 1000,
    render: bool = False,
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
    number_agents = 4
    n = number_agents * 2
    env = gym.make("multi-point-robots-vel-v0", render=render, dt=0.01, number_agents=number_agents)
    _ = env.reset()
    init_pos = np.random.random(n)
    init_vel = np.random.random(n)
    action = env.action_space.sample()
    print("Starting episode")
    observation_history = []
    for i in range(n_steps):
        #action = time_variant_action(env.t(), 12)
        ob, _, _, _ = env.step(action)
        observation_history.append(ob)
        if i % 100 == 0:
            print(f"ob : {ob}")
    return observation_history


if __name__ == "__main__":
    run_multi_point_robots(render=True)
