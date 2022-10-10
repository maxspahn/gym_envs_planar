# pylint: disable=import-outside-toplevel
import gym
import planarenvs.mobile_base  # pylint: disable=unused-import
import numpy as np


def run_mobile_base(render=False, n_steps=1000, obstacles=False, goal=False):
    """
    Minimal example of mobile base.

    The mobile base robot is a block that operates on a one-dimensional plane.
    It has thus one degree of freedom. The observation contains the current
    position and current velocity of the robot:
        x: [`x`]
        xdot: [`xdot`].
    """
    env = gym.make("mobile-base-vel-v0", render=render, dt=0.01)
    action = [0.4]
    ob = env.reset(pos=np.array([-2.0]), vel=np.array([0.5]))
    if obstacles:
        from planarenvs.scenes.obstacles import (
            sphereObst1,
            sphereObst2,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
    print("Starting episode")
    history = []
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        if i % 100 == 0:
            print(f"ob : {ob}")
        history.append(ob)
    return history


if __name__ == "__main__":
    run_mobile_base(render=True)
