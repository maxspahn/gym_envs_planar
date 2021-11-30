from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal

goal1Dict = {
    "m": 2, "w": 1.0, "prime": True, 'indices': [0, 1], 'parent_link': 0, 'child_link': 3,
    'desired_position': [1, 2], 'epsilon': 0.2, 'type': "staticSubGoal", 
}

goal1 = StaticSubGoal(name="goal1", contentDict=goal1Dict)
