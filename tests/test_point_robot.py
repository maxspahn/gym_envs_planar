import gym
import planarenvs.pointRobot
import numpy as np
import pytest

@pytest.fixture
def pointRobotEnv():
    initPos = np.array([0.0, -1.0])
    initVel = np.array([-1.0, 0.0])
    env = gym.make("point-robot-vel-v0", render=False, dt=0.01)
    _ = env.reset(pos=initPos, vel=initVel)
    return env, initPos, initVel


def test_pointRobot(pointRobotEnv):
    env, initPos, initVel = pointRobotEnv
    action = np.zeros(2)
    ob, reward, done, info = env.step(action)
    assert isinstance(ob, dict)
    assert isinstance(ob['x'], np.ndarray)
    assert ob['x'].size == 2
    np.testing.assert_array_equal(ob['x'], initPos)


