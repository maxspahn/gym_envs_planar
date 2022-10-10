import pytest


def blueprint_test(test_main):
    """
    Blueprint for environment tests.
    An environment main always has the four arguments:
        - n_steps: int
        - render: bool
        - goal: bool
        - obstacles: bool

    The function verifies if the main returns a list of observations.
    """
    history = test_main(n_steps=100, render=False, goal=True, obstacles=True)
    assert isinstance(history, list)


def test_point_robot():
    from point_robot import run_point_robot
    blueprint_test(run_point_robot)


def test_mobile_robot():
    from mobile_robot import run_mobile_robot
    blueprint_test(run_mobile_robot)


def test_n_link_reacher():
    from n_link_reacher import run_n_link_reacher
    blueprint_test(run_n_link_reacher)

def test_ground_robot():
    from ground_robot import run_ground_robot
    blueprint_test(run_ground_robot)

def test_ground_robot():
    from ground_robot_arm import run_ground_robot_arm
    blueprint_test(run_ground_robot_arm)

def test_mobile_base():
    from mobile_base import run_mobile_base
    blueprint_test(run_mobile_base)
