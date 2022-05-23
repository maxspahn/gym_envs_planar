# pylint: disable=import-outside-toplevel
import gym
import planarenvs.mobile_robot  # pylint: disable=unused-import
import numpy as np


def run_mobile_robot(
    n_steps=1000, render=False, goal: bool = False, obstacles: bool = False
):
    """
    Minimal example of mobile robot with (n-1) arm joints.

    The mobile robot is a block with a n-degrees of freedom arm attached.
    It operates in the two-dimensional plane.
    It has thus n degrees of freedom. The observation contains the current
    position and current velocity of the robot:
        x: [`x`, `q`]
        xdot: [`xdot`, `qdot`].
    """
    n = 4
    env = gym.make("mobile-robot-acc-v0", render=render, n=n, dt=0.01)
    action = np.zeros(n)
    action[0] = 1.0
    action[3] = 1.0
    ob = env.reset(pos=np.random.rand(n))
    if obstacles:
        from examples.obstacles import (
            sphereObst1,
            sphereObst2,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
    print("Starting episode")
    observation_history = []
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        observation_history.append(ob)
        if i % 100 == 0:
            print(f"ob : {ob}")
    return observation_history


if __name__ == "__main__":
    obstacles = False
    run_mobile_robot(n_steps=1000, render=True, obstacles=obstacles)
