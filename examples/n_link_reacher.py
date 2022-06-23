# pylint: disable=import-outside-toplevel
import gym
import planarenvs.n_link_reacher  # pylint: disable=unused-import
import numpy as np


def run_n_link_reacher(
    n_steps=1000,
    render: bool = True,
    goal: bool = False,
    obstacles: bool = False,
):
    """
    Minimal example for n-link planar robot arm.

    The n-link-arm is a n-degrees of freedom robotic arm operating in the
    two-dimensional plane. In a sense, it is extended pendulum. The observation
    is the state of the joints:
        x: [`q`]
        xdot: [`qdot`]
    """
    n = 3
    env = gym.make("nLink-reacher-vel-v0", render=render, n=n, dt=0.01)
    action = np.ones(n) * 8 * 0.01
    ob = env.reset(pos=np.random.rand(n))
    if obstacles:
        from planarenvs.scenes.obstacles import (
            sphereObst1,
            sphereObst2,
            dynamicSphereObst2,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(dynamicSphereObst2)
    if goal:
        from planarenvs.scenes.goal import (
            splineGoal,
        )

        env.add_goal(splineGoal)
    print("Starting episode")
    observation_history = []
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        observation_history.append(ob)
        if i % 100 == 0:
            print(f"ob : {ob}")
    return observation_history


if __name__ == "__main__":
    obstacles = True
    goal = True
    run_n_link_reacher(goal=goal, obstacles=obstacles)
