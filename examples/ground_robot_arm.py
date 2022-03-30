import gym
import planarenvs.ground_robots # pylint: disable=unused-import
import numpy as np

obstacles = True


def main():
    env = gym.make("ground-robot-arm-vel-v0", render=True, dt=0.01)
    defaultAction = np.array([1.1, 0.50, 0.2])
    # env = gym.make('ground-robot-vel-v0', render=True, dt=0.01)
    # defaultAction = np.array([1.0, 0.0])
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
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
        # ob = env.reset(pos=np.array([0.0, 1.0, 0.6 * np.pi]), vel=np.array([0.1, 0.0]))
        print("Starting episode")
        for i in range(n_steps):
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward
            if done:
                break


if __name__ == "__main__":
    main()
