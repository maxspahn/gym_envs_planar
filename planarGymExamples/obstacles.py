from MotionPlanningEnv.sphereObstacle import SphereObstacle

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
