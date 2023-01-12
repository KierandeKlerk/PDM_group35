"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import sys
import os
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from tqdm import tqdm

import pybullet as p
import time
import pkg_resources
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from quadrotor_project.planningAlgorithms.RRT import GridRRTstar2D, GridRRTstar3D
import quadrotor_project.planningAlgorithms.occupancyGridTools as GT
from quadrotor_project.envs.RRTAviary import RRTAviary

########################################################
'''
Choose a track: 
- 0 for a sample trajectory (spiral)
- 1 for a simple course with obstacle variance in x,y plane, path generated by 2D RRT* algorithm
- 2 for a (relatively) simple course with obstacle variance in 3D, path generated by 3D RRT* algorithm
- 3 for a complex course with obstacle variance in 3D, it is strongly adivsedto set 'loadpath' to True (generating the path from scratch takes about 4 hours of computing on a relatively high end laptop)
'''
track = 3

loadPath = False
savePath = True
########################################################




drone = DroneModel("cf2x")
num_drones = 1 # Do not change
physics = Physics("pyb")
#vision = True
gui = True
plot = True
user_debug_gui = False
aggregate = True
obstacles = True # Turn to True for default arena
record_video = False
use_logger = False
save_logger = False
do_print = False

# Simulation settings
simulation_freq_hz = 240
control_freq_hz = 48
duration_sec = 65
output_folder = 'results'
colab = False

#############################################################
#### Initialize the simulation ##############################
#############################################################
AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

#### Create the environment 

env = RRTAviary(drone_model=drone,
                num_drones=num_drones,
                physics=physics,
                neighbourhood_radius=10,
                freq=simulation_freq_hz,
                aggregate_phy_steps=AGGR_PHY_STEPS,
                gui=gui,
                record=record_video,
                obstacles=obstacles,
                track = track,
                load_path = loadPath,
                save_path = savePath,
                user_debug_gui=user_debug_gui
                )
INIT_XYZS = env.INIT_XYZS
INIT_RPYS = env.INIT_RPYS

### Generate waypoints
path_refit, track_time = env.getPath()
numWP = int(track_time*control_freq_hz)
ind = np.round(np.linspace(0, len(path_refit) - 1, numWP)).astype(int)
fine_waypoints = path_refit[ind,:]
wp_counter = 0

#### Initialize the logger #################################
if use_logger:
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

#### Initialize the controllers ############################
ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]


#### Run the simulation ####################################
CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
action = {str(i): np.array([0,0,0,0]) for i in range(num_drones)}
START = time.time()

# Static cameras
# p.resetDebugVisualizerCamera(0.01, -89.99,-89.99,[0,3,5]) # turn on to get view from above (static)
# p.resetDebugVisualizerCamera(3, -45,-45,[0,1,1]) # turn on to get view from above (static)
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[2.5, 2.5, 2.5])

for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
    
    #### Step the simulation ###################################
    obs, reward, done, info = env.step(action)
    
    #### Compute control at the desired frequency ##############
    if i%CTRL_EVERY_N_STEPS == 0:
        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                    state=obs[str(j)]["state"],
                                                                    target_pos=fine_waypoints[wp_counter],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )
            wp_counter += 1 if wp_counter<numWP-1 else 0
        yaw, pitch, roll = p.getEulerFromQuaternion(obs[str(j)]["state"][3:7])
        # p.resetDebugVisualizerCamera(0.001, 0, -roll*57.2958-30,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone POV
        # p.resetDebugVisualizerCamera(0.25, -90, -30,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone from behind
        # p.resetDebugVisualizerCamera(0.6, 0, -70,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone from above

        # print(f'Time right now: {i/env.SIM_FREQ}')

    #### Log the simulation ####################################
    if use_logger:
        for j in range(num_drones):
            logger.log(drone=j,
                    timestamp=i/env.SIM_FREQ,
                    state=obs[str(j)]["state"],
                    control=np.hstack([fine_waypoints[wp_counter, 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                    )

    #### Printout ##############################################
    if do_print:
        if i%env.SIM_FREQ == 0:
            env.render()

    #### Sync the simulation ###################################
    if gui:
        sync(i, START, env.TIMESTEP)

#### Close the environment #################################
env.close()

#### Save the simulation results ###########################
if save_logger:
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

# #### Plot the simulation results ###########################
if plot and use_logger:
    logger.plot()