import gym
import nLinkReacher
import numpy as np

obstacles = False
goal = False


def main():
    n = 6
    # env = gym.make('nLink-reacher-acc-v0', n=n, dt=0.01)
    env = gym.make("nLink-reacher-vel-v0", render=True, n=n, dt=0.01)
    defaultAction = np.ones(n) * 8 * 0.01
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=np.random.rand(n))
        if obstacles:
            from planarGymExamples.obstacles import sphereObst1, sphereObst2

            env.addObstacle(sphereObst1)
            env.addObstacle(sphereObst2)
        if goal:
            from planarGymExamples.goal import goal1

            env.addGoal(goal1)
        print("Starting episode")
        for i in range(n_steps):
            action = env.action_space.sample()
            action = defaultAction
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == "__main__":
    main()
