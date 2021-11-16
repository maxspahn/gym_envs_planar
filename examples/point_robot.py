import gym
import pointRobot
import numpy as np


def main():
    env = gym.make('point-robot-vel-v0', render=True, dt=0.01)
    defaultAction = np.array([-0.2, 0.10])
    defaultAction = lambda t: np.array([np.cos(1.0 * t), np.sin(1.0 * t)])
    initPos = np.array([0.0, -1.0])
    initVel = np.array([-1.0, 0.0])
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=initPos, vel=initVel)
        print("Starting episode")
        t = 0
        for i in range(n_steps):
            t = t + env.dt()
            action = env.action_space.sample()
            action = defaultAction(t)
            ob, reward, done, info = env.step(action)
            cumReward += reward
            if done:
                break


if __name__ == '__main__':
    main()
