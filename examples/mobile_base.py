import gym
import mobileBase
import numpy as np

obstacles = False


def main():
    env = gym.make("mobile-base-acc-v0", render=True, dt=0.01)
    defaultAction = [0.4]
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=np.array([-2.0]), vel=np.array([0.5]))
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
