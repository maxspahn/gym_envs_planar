import gym
import planarenvs.mobile_robot  # pylint: disable=unused-import
import numpy as np

obstacles = False


def main():
    n = 5
    env = gym.make("mobile-robot-acc-v0", render=True, n=n, dt=0.01)
    action = np.zeros(n)
    action[0] = 1.0
    action[3] = 1.0
    n_steps = 400
    ob = env.reset(pos=np.random.rand(n))
    if obstacles:
        from examples.obstacles import (
            sphereObst1,
            sphereObst2,
        )  # pylint: disable=import-outside-toplevel

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
    print("Starting episode")
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        print(f"ob : {ob}")


if __name__ == "__main__":
    main()
