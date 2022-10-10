# pylint: disable=import-outside-toplevel
import gym
import planarenvs.ground_robots  # pylint: disable=unused-import
import numpy as np

def run_ground_robot(n_steps=1000, render=False, obstacles=False, goal=False):
    """
    Minimal example for ground robot environment.

    The ground robot is a differential drive for which the actuation is on the
    forward and angular velocity or acceleration. The observation is composed
    of the position and velocity in the workspace and the current forward and
    angular velocity under the key `vel`:
        [`joint_position`][`position`]: [`x`, `y`, `theta`]
        [`joint_position`][`velocity`]: [`xdot`, `ydot`, `thetadot`]
        [`joint_position`][`forward_velocity`]: [`forward_velocity`, `angular_velocity`]
    """
    env = gym.make("ground-robot-vel-v0", render=render, dt=0.01)
    action = np.array([1.0, 1.00])
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.6 * np.pi, 0.5]),
        vel=np.array([0.1, 0.0, 0.1]),
    )
    if obstacles:
        from planarenvs.scenes.obstacles import (
            sphereObst1,
            sphereObst2,
            dynamicSphereObst1,
        )

        env.add_obstacle(sphereObst1)
        env.add_obstacle(sphereObst2)
        env.add_obstacle(dynamicSphereObst1)
    history = []
    for i in range(n_steps):
        ob, _, _, _ = env.step(action)
        if i % 100 == 0:
            print(f"ob : {ob}")
        history.append(ob)
    return history


if __name__ == "__main__":
    run_ground_robot(render=True)
