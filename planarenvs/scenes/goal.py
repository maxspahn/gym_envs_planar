from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpscenes.goals.dynamic_sub_goal import DynamicSubGoal

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
    "is_primary_goal": False,
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
    "trajectory": ["1 + sp.cos(0.3 * t) + 1", "2.0"],
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
