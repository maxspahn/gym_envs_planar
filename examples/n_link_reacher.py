#pylint: disable=import-outside-toplevel
import gym
import planarenvs.n_link_reacher #pylint: disable=unused-import
import numpy as np

obstacles = False
goal = True


def main():
    """
    Minimal example for n-link planar robot arm.

    The n-link-arm is a n-degrees of freedom robotic arm operating in the
    two-dimensional plane. In a sense, it is extended pendulum. The observation
    is the state of the joints:
        x: [`q`]
        xdot: [`qdot`]
    """
    n = 3
    env = gym.make("nLink-reacher-vel-v0", render=True, n=n, dt=0.01)
    action = np.ones(n) * 8 * 0.01
    n_steps = 100000
    ob = env.reset(pos=np.random.rand(n))
    if obstacles:
        from examples.obstacles import (
            sphereObst1,
            sphereObst2,
            dynamicSphereObst2,
        )
        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(dynamicSphereObst2)
    if goal:
        from examples.goal import (
            staticGoal,
        )
        env.add_goal(staticGoal)
    print("Starting episode")
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        print(f"ob : {ob}")


if __name__ == "__main__":
    main()
