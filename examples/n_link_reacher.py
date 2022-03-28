import gym
import planarenvs.n_link_reacher  # pylint: disable=unused-import
import numpy as np

obstacles = True
goal = True


def main():
    n = 3
    # env = gym.make('nLink-reacher-acc-v0', n=n, dt=0.01)
    env = gym.make("nLink-reacher-vel-v0", render=True, n=n, dt=0.01)
    action = np.ones(n) * 8 * 0.01
    n_steps = 1000
    ob = env.reset(pos=np.random.rand(n))
    if obstacles:
        from examples.obstacles import (
            sphereObst1,
            sphereObst2,
            dynamicSphereObst2,
        )  # pylint: disable=import-outside-toplevel

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(dynamicSphereObst2)
    if goal:
        from examples.goal import (
            splineGoal,
        )  # pylint: disable=import-outside-toplevel

        env.add_goal(splineGoal)
    print("Starting episode")
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        print(f"ob : {ob}")


if __name__ == "__main__":
    main()
