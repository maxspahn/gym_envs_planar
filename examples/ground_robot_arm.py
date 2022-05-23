# pylint: disable=import-outside-toplevel
import gym
import planarenvs.ground_robots  # pylint: disable=unused-import
import numpy as np

obstacles = True


def main():
    """
    Minimal example for ground robot with arm environment.

    The ground robot is a differential drive for which the actuation is on the
    forward and angular velocity or acceleration. The observation is composed
    of the position and velocity in the workspace for the base and the joint
    position and velocity for the arm. Additionally, the current forward and
    angular velocity of the base are returned under the key `vel`:
        x: [`x`, `y`, `theta`, `q`]
        xdot: [`xdot`, `ydot`, `thetadot`, `qdot`]
        vel: [`forward_velocity`, `angular_velocity`]
    """
    env = gym.make("ground-robot-arm-vel-v0", render=True, dt=0.01)
    action = np.array([1.1, 0.50, 0.2])
    n_steps = 1000
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.6 * np.pi, 0.5]),
        vel=np.array([0.1, 0.0, 0.1]),
    )
    if obstacles:
        from examples.obstacles import (
            sphereObst1,
            sphereObst2,
            dynamicSphereObst1,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(dynamicSphereObst1)
    print("Starting episode")
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        if i % 100 == 0:
            print(f"ob : {ob}")


if __name__ == "__main__":
    main()
