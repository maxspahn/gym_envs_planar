# pylint: disable=import-outside-toplevel
import gym
import planarenvs.mobile_base  # pylint: disable=unused-import
import numpy as np

obstacles = False


def main():
    """
    Minimal example of mobile base.

    The mobile base robot is a block that operates on a one-dimensional plane.
    It has thus one degree of freedom. The observation contains the current
    position and current velocity of the robot:
        x: [`x`]
        xdot: [`xdot`].
    """
    env = gym.make("mobile-base-vel-v0", render=True, dt=0.01)
    action = [0.4]
    n_steps = 1000
    ob = env.reset(pos=np.array([-2.0]), vel=np.array([0.5]))
    if obstacles:
        from examples.obstacles import (
            sphereObst1,
            sphereObst2,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
    print("Starting episode")
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        if i % 100 == 0:
            print(f"ob : {ob}")


if __name__ == "__main__":
    main()
