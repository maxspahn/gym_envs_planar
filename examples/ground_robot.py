import gym
import groundRobots
import numpy as np


def main():
    env = gym.make('ground-robot-diffdrive-vel-v0', dt=0.01)
    defaultAction = np.array([1.0, 0.050])
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        #ob = env.reset(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
        ob = env.reset(np.array([0.0, 1.0, 0.0]))
        print("Starting episode")
        for i in range(n_steps):
            # action = env.action_space.sample()
            action = defaultAction
            env.render()
            ob, reward, done, info = env.step(action)
            cumReward += reward
            if done:
                break


if __name__ == '__main__':
    main()
