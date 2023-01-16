# PDM group35's repository
This is the repository for the Planning & Decision-Making (RO47005) final project by group 35. </br>
The aim of this repository is to provide simulations of a quadrotor navigating through various tracks by using RRT* as a path planning algorithm and PID for local control. These simulations were done for 2 different models: one where the quadrotor is modelled as a point mass with its thrust modelled a single vector originating from this point, another used gym-pybullet-drones's model where the thrust and torque exerted by each individual motor is taken computed and controlled.

# Contents and structure
For ease of installation, [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones.git) is included as a submodule. </br>
Our own files are all stored under the PDM_project/ folder. Here, the 2 main simulation files can be found: [pointmass.py](PDM_project/pointmass.py) and [completeModel.py](PDM_project/completeModel.py). 
## Quadrotor project package
Our custom package, [quadrotor_project](PDM_project/quadrotor_project/), offers multiple subdirectories:
- [assets/](PDM_project/quadrotor_project/assets/) : contains obstacle mesh and urdf files and precomputed paths as npy files
- [control/](PDM_project/quadrotor_project/control/) : contains a (PID) control class.
- [envs/](PDM_project/quadrotor_project/envs/) : contains environment classes inspired by or derived from [BaseAviary](gym-pybullet-drones/gym_pybullet_drones/envs/BaseAviary.py) and [CtrlAviary](gym-pybullet-drones/gym_pybullet_drones/envs/CtrlAviary.py)
- [planningAlgorithms/](PDM_project/quadrotor_project/planningAlgorithms/) : contains occupancy grid functions and 2 RRT* classes (2D and 3D). 

## [pointmass.py](PDM_project/pointmass.py)
This file simulates the quadrotor as a point mass in [track 1](PDM_project/quadrotor_project/assets/track1.urdf), taking a 2D slice of the generated occupancy grid (see [occupancyGridTools.py](PDM_project/quadrotor_project/planningAlgorithms/occupancyGridTools.py)) to feed it into the GridRRTstar2D (found in [RRT.py](PDM_project/quadrotor_project/planningAlgorithms/RRT.py)). The path generated is then followed by applying PID control (see [pointmassPID.py](PDM_project/quadrotor_project/control/pointmassPID.py)).

## [completeModel.py](PDM_project/completeModel.py)
This file simulates [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones.git)'s quadrotor model in any of the provided tracks (see the track selection in the file). The path is again computed from an occupancy grid and fed into an RRT* algorithm (2D or 3D, depending on the chosen track).

## RRT demos
### [MainRRT.py](PDM_project/RRT/MainRRT.py)
Hier aub thissoe
### [VideoRRT.py](PDM_project/RRT//VideoRRT.py)
en hier ook @thijsdomurg

# Installation guide
This guide is exclusively for installation on Linux based operating systems, though you might be able to adapt this guide's steps for other systems.</br>
Firstly, we'll have to clone the repository; open your command terminal and execute the following commands:</br>
```
$ cd path/to/desired/folder
$ git clone https://github.com/KierandeKlerk/PDM_group35.git
```
Next we'll 


### Authors:
 - Fabian Gebben 
 - Thijs Domburg
 - Amin Mimoun Bouras
 - Kieran de Klerk





# TO DO
- Add installation instructions(README)
- Add more convenient camera selection

when you get a ModuleNotFoundError for quadrotor_project, try export PYTHONPATH="${PYTHONPATH}:/path/to/project/" where the project in question is the folder containing the file generating the error.

When running completeModel.py or pointmass.py and recomputing a path, the program pauses while a plot is showing

