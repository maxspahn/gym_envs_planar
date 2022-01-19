### Generic Planar Robots

In this package, generic robot for simple tests are made available for gym environments.

## Dependencies

This package depends on casadi for dynamics generation and gym.
Dependencies will be installed through pip installation, see below.
It uses the lightweight implementation for forward kinematics in casadi.



## Installation
Clone the repository 
```bash
git clone git@github.com:maxspahn/gym_envs_planar.git
```
### virtual environment installation using poetry
If you are not familiar with poetry see [Poetry Installation](https://python-poetry.org/docs/).
Once poetry is installed you can run
```bash
poetry install
```
If you want the motion planning scenes
```bash
poetry install -E scenes
```
The virtual environment with everything installed is entered with
```bash
poetry shell
```

### global installation via pip
```bash
pip3 install .
```

When obstacles are required, you must use
```bash
pip3 install ".[scenes]"
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
