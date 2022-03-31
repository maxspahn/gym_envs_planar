import gym
import planarenvs.point_robot
import planarenvs.n_link_reacher
import planarenvs.mobile_base
import planarenvs.mobile_robot
import numpy as np
import pytest

@pytest.fixture
def pointRobotEnv():
    init_pos = np.array([0.0, -1.0])
    init_vel = np.array([-1.0, 0.0])
    env = gym.make("point-robot-acc-v0", render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return env, init_pos, init_vel

@pytest.fixture
def nLinkReacherEnv():
    n = 5
    init_pos = np.zeros(n)
    init_vel = np.zeros(n)
    init_pos[0] = 0.2
    init_pos[1] = 0.4
    env = gym.make("nLink-reacher-acc-v0", n=n, render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return env, init_pos, init_vel

@pytest.fixture
def mobileBaseEnv():
    init_pos = np.array([-1.0])
    init_vel = np.array([0.2])
    env = gym.make('mobile-base-acc-v0', render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return env, init_pos, init_vel

@pytest.fixture
def mobileRobotEnv():
    n = 5
    init_pos = np.zeros(n)
    init_vel = np.zeros(n)
    init_pos[0] = 0.2
    init_pos[1] = 0.4
    env = gym.make('mobile-robot-acc-v0', n=n, render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)
    return env, init_pos, init_vel

@pytest.fixture
def allEnvs(pointRobotEnv, nLinkReacherEnv, mobileBaseEnv, mobileRobotEnv):
    return list(locals().values())


def test_all(allEnvs):
    for env in allEnvs:
        ob = env[0].reset(pos=env[1], vel=env[2])
        action = np.random.random(env[0].n)
        np.testing.assert_array_almost_equal(ob['x'], env[1], decimal=2)
        ob, _, _, _ = env[0].step(action)
        assert isinstance(ob, dict)
        assert isinstance(ob['x'], np.ndarray)
        assert isinstance(ob['xdot'], np.ndarray)
        assert ob['x'].size == env[0].n
        assert ob['xdot'].size == env[0].n


