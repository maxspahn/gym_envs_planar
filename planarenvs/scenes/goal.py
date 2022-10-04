from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal

staticGoalDict = {
    "weight": 1.0,
    "is_primary_goal": True,
    "indices": [0, 1],
    "parent_link": 0,
    "child_link": 3,
    "desired_position": [1, 2],
    "epsilon": 0.2,
    "type": "staticSubGoal",
}

staticGoal = StaticSubGoal(name="goal1", content_dict=staticGoalDict)
lineGoalDict = {
    "weight": 1.0,
    "is_primary_goal": True,
    "indices": [1],
    "parent_link": 0,
    "child_link": 3,
    "desired_position": [0.5],
    "angle": 0.5,
    "epsilon": 0.2,
    "type": "staticSubGoal",
}

lineGoal = StaticSubGoal(name="goal2", content_dict=lineGoalDict)
analyticGoalDict = {
    "weight": 1.0,
    "is_primary_goal": True,
    "indices": [0, 1],
    "parent_link": 0,
    "child_link": 3,
    "trajectory": ["1 + ca.cos(0.3 * t)", "2"],
    "epsilon": 0.2,
    "type": "analyticSubGoal",
}
analyticGoal = DynamicSubGoal(name="goal2", content_dict=analyticGoalDict)
splineDict = {
    "degree": 2,
    "controlPoints": [[-2.0, 1.0], [2.0, 0.0], [4.0, 2.0], [3.0, 2.0]],
    "duration": 10,
}
splineGoalDict = {
    "weight": 1.0,
    "is_primary_goal": True,
    "indices": [0, 1],
    "parent_link": 0,
    "child_link": 3,
    "trajectory": splineDict,
    "epsilon": 0.2,
    "type": "splineSubGoal",
}
splineGoal = DynamicSubGoal(name="goal2", content_dict=splineGoalDict)
