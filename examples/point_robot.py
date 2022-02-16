import gym
import planarenvs.pointRobot
import numpy as np


obstacles = True
goal = True
sensors = True


def main():
    env = gym.make("point-robot-vel-v0", render=True, dt=0.01)
    defaultAction = np.array([-0.2, 0.10])
    defaultAction = lambda t: np.array([np.cos(1.0 * t), np.sin(1.0 * t)])
    initPos = np.array([0.0, -1.0])
    initVel = np.array([-1.0, 0.0])
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset(pos=initPos, vel=initVel)
        env.resetLimits(pos={'high': np.array([1.0, 3.0]), 'low': np.array([-1.0, -3.0])})

        if sensors:
            from sensors.GoalSensor import GoalSensor
            from sensors.ObstacleSensor import ObstacleSensor

            obstSensorPos = ObstacleSensor(nbObstacles=2, mode='position')
            env.addSensor(obstSensorPos)
            obstSensorDist = ObstacleSensor(nbObstacles=2, mode='distance')
            env.addSensor(obstSensorDist)
            goalDistObserver = GoalSensor(nbGoals=1, mode='distance')
            env.addSensor(goalDistObserver)
            goalPosObserver = GoalSensor(nbGoals=1, mode='position')
            env.addSensor(goalPosObserver)

        if obstacles:
            from examples.obstacles import sphereObst1, sphereObst2, dynamicSphereObst1, dynamicSphereObst2

            env.addObstacle(sphereObst1)
            env.addObstacle(sphereObst2)
            env.addObstacle(dynamicSphereObst1)
            env.addObstacle(dynamicSphereObst2)
        if goal:
            from examples.goal import splineGoal

            env.addGoal(splineGoal)
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


if __name__ == "__main__":
    main()
