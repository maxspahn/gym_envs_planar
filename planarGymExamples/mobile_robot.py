import gym
import mobileRobot
import numpy as np

obstacles = False


def main():
    n = 5
    env = gym.make("mobile-robot-acc-v0", render=True, n=n, dt=0.01)
    defaultAction = np.zeros(n)
    defaultAction[0] = 1.0
    defaultAction[3] = 1.0
    n_episodes = 1
    n_steps = 400
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=np.random.rand(n))
        if obstacles:
            from planarGymExamples.obstacles import sphereObst1, sphereObst2

            env.addObstacle(sphereObst1)
            env.addObstacle(sphereObst2)
        print("Starting episode")
        for i in range(n_steps):
            action = env.action_space.sample()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward
            if done:
                break


if __name__ == "__main__":
    main()
