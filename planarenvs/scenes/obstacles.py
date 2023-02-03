from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

obst1Dict = {
    "type": "sphere",
    "geometry": {"position": [0.0, 5.0], "radius": 1.0},
}
sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
obst2Dict = {
    "type": "sphere",
    "geometry": {"position": [2.0, -1.0], "radius": 0.2},
}
sphereObst2 = SphereObstacle(name="simpleSphere", content_dict=obst2Dict)
dynamicObst1Dict = {
    "type": "analyticSphere",
    "geometry": {"trajectory": ["1.1 * t", "-2.0 + 0.1 * t"], "radius": 0.2},
}
dynamicSphereObst1 = DynamicSphereObstacle(
    name="dynamicSphere", content_dict=dynamicObst1Dict
)
splineDict = {
    "degree": 2,
    "controlPoints": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]],
    "duration": 10,
}
dynamicObst2Dict = {
    "type": "splineSphere",
    "geometry": {"trajectory": splineDict, "radius": 0.2},
}
dynamicSphereObst2 = DynamicSphereObstacle(
    name="dynamicSphere", content_dict=dynamicObst2Dict
)
