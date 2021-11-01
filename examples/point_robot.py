import gym
import pointRobot
import numpy as np


def main():
    env = gym.make('point-robot-acc-v0', render=True, dt=0.01)
    defaultAction = [0.0, 0.10]
    initPos = np.array([1.0, 0.0])
    initVel = np.array([-1.0, 0.0])
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(initPos, initVel)
        print("Starting episode")
        for i in range(n_steps):
            action = env.action_space.sample()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward
            if done:
                break


if __name__ == '__main__':
    main()
