from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle

obst1Dict = {
    "dim": 2,
    "type": "sphere",
    "geometry": {"position": [0.0, 5.0], "radius": 1.0},
}
sphereObst1 = SphereObstacle(name="simpleSphere", contentDict=obst1Dict)
obst2Dict = {
    "dim": 2,
    "type": "sphere",
    "geometry": {"position": [2.0, -1.0], "radius": 0.2},
}
sphereObst2 = SphereObstacle(name="simpleSphere", contentDict=obst2Dict)
dynamicObst1Dict = {
    "dim": 2,
    "type": "analyticSphere",
    "geometry": {"trajectory": ["1.1 * t", "-2.0 + 0.1 * t"], "radius": 0.2},
}
dynamicSphereObst1 = DynamicSphereObstacle(
    name="dynamicSphere", contentDict=dynamicObst1Dict
)
splineDict = {
    "degree": 2,
    "controlPoints": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]],
    "duration": 10,
}
dynamicObst2Dict = {
    "dim": 2,
    "type": "splineSphere",
    "geometry": {"trajectory": splineDict, "radius": 0.2},
}
dynamicSphereObst2 = DynamicSphereObstacle(
    name="dynamicSphere", contentDict=dynamicObst2Dict
)
