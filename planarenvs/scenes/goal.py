from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal

staticGoalDict = {
    "m": 2,
    "w": 1.0,
    "prime": True,
    "indices": [0, 1],
    "parent_link": 0,
    "child_link": 3,
    "desired_position": [1, 2],
    "epsilon": 0.2,
    "type": "staticSubGoal",
}

staticGoal = StaticSubGoal(name="goal1", contentDict=staticGoalDict)
lineGoalDict = {
    "m": 1,
    "w": 1.0,
    "prime": True,
    "indices": [1],
    "parent_link": 0,
    "child_link": 3,
    "desired_position": [0.5],
    "angle": 0.5,
    "epsilon": 0.2,
    "type": "staticSubGoal",
}

lineGoal = StaticSubGoal(name="goal2", contentDict=lineGoalDict)
analyticGoalDict = {
    "m": 2,
    "w": 1.0,
    "prime": True,
    "indices": [0, 1],
    "parent_link": 0,
    "child_link": 3,
    "trajectory": ["1 + ca.cos(0.3 * t)", "2"],
    "epsilon": 0.2,
    "type": "analyticSubGoal",
}
analyticGoal = DynamicSubGoal(name="goal2", contentDict=analyticGoalDict)
splineDict = {
    "degree": 2,
    "controlPoints": [[-2.0, 1.0], [2.0, 0.0], [4.0, 2.0], [3.0, 2.0]],
    "duration": 10,
}
splineGoalDict = {
    "m": 2,
    "w": 1.0,
    "prime": True,
    "indices": [0, 1],
    "parent_link": 0,
    "child_link": 3,
    "trajectory": splineDict,
    "epsilon": 0.2,
    "type": "splineSubGoal",
}
splineGoal = DynamicSubGoal(name="goal2", contentDict=splineGoalDict)
