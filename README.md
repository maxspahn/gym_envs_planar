### Generic Planar Robots

In this package, generic robot for simple tests are made available for gym environments.

## Dependencies

This package depends on casadi for dynamics generation and gym.
Dependencies will be installed through pip installation, see below.
It uses the lightweight implementation for forward kinematics in casadi.


When obstacles are required, you must use
```bash
pip3 install -r requirements_scenes.txt
```

## Installation
```bash
pip3 install -e .
```

## Switching

Environments can be created using the normal gym syntax.
For example the below code line creates a planar robot with 3 links and a constant k.
Actions are torques to the individual joints.
```python
env = gym.make('nLink-reacher-tor-v0', n=3, dt=0.01, k=2.1)
```
## Examples

For a constant controlled torque, the simulation is displayed below:

 
![Example of torque controlled environment](./assets/torques.gif) 
