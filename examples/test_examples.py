import pytest

from point_robot import main as point_robot_main
from mobile_robot import main as mobile_robot_main


def test_point_robot():
    point_robot_main()

def test_mobile_robot():
    mobile_robot_main()

