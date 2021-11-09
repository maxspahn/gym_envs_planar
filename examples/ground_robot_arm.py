import gym
import groundRobots
import numpy as np


def main():
    env = gym.make('ground-robot-diffdrive-arm-acc-v0', render=True, dt=0.01)
    defaultAction = np.array([0.05, 0.0, 0.0])
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=[0.0, 0.0, 0.0, 0.1], vel=[0.0, 0.0, 0.0])
        print("Starting episode")
        for i in range(n_steps):
            action = defaultAction
            #env.render()
            ob, reward, done, info = env.step(action)
            cumReward += reward
            if done:
                break


if __name__ == '__main__':
    main()
