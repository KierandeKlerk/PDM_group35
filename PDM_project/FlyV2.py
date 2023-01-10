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
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from tqdm import tqdm

import pybullet as p
import time
import pybullet_data
import pkg_resources
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from quadrotor_project.planningAlgorithms.RRT import GridRRTstar2D, GridRRTstar3D
import quadrotor_project.planningAlgorithms.occupancyGridTools as GT

drone = DroneModel("cf2x")
num_drones = 1
physics = Physics("pyb")
vision = False
gui = True
record_vision = False
plot = True
user_debug_gui = False
aggregate = True
obstacles = False # Turn to True for default arena
record_video = False
use_logger = False


# Simulation settings
simulation_freq_hz = 240
control_freq_hz = 48
duration_sec = 20
control_duration = 15 
output_folder = 'results'
colab = False
initial_position = np.array([0.5, 0.5, 0.5])
goal_position = np.array([6, 2, 2])
tracking_dimension = 3 # 2 or 3 or None

if tracking_dimension == 2:
    obstacleToAdd = pkg_resources.resource_filename("quadrotor_project", "assets/drone_parcours.obj")
elif tracking_dimension == 3: 
    obstacleToAdd = pkg_resources.resource_filename("quadrotor_project", "assets/Hole_obstacle_course.obj")

# Grid settings
grid_pitch = 0.05 
dgoal = 5
dsearch = 10
dcheaper = 15
max_iter = 10000
track_time = 18
loadPath = False
savePath = True
relativepath = "PDM/Project/PDM_group35/PDM_project/"


# Generate occupancy grid
if tracking_dimension == 2:
    if not loadPath:
        # Generating occupancy grid
        occupancyGrid, _ = GT.generateOccupancyGrid(obstacleToAdd)
        
        # Applying RRT*
        start = (initial_position[:2])/grid_pitch
        goal = (goal_position[:2])/grid_pitch
        grid2D = occupancyGrid[:,:,0]
        marginGrid2D = GT.marginWithDepth(grid2D, desiredMarginDepthinMeters=0.2, pitchInMeters=grid_pitch)
        graph = GridRRTstar2D(start, goal, dgoal, dsearch, dcheaper, grid2D, marginGrid2D, max_iter)
        
        starttime = time.time()
        pbar = tqdm(total = max_iter)
        graph.makemap()
        while graph.iterations():
            graph.expand()
            pbar.update(1)
        endtime = time.time()
        pbar.close()
        print("Time elapsed: ", endtime - starttime)
        graph.makemap()
        # Convert path to simulation scale and frame
        path = np.array(graph.smoothpath).T
        path = np.append(path,initial_position[2]/grid_pitch*np.ones((len(path),1)), axis = 1)
        path_refit = path*grid_pitch
        if savePath:
            np.save(os.path.join(relativepath,"smoothpath2D.npy"), path_refit)
    else: 
        path_refit = np.load(os.path.join(relativepath,"smoothpath2D.npy"))
elif tracking_dimension == 3:
    if not loadPath:
        # Generating occupancy grid
        occupancyGrid, _ = GT.generateOccupancyGrid(obstacleToAdd)
        
        # Applying RRT*
        start = initial_position/grid_pitch
        goal = goal_position/grid_pitch
        grid3D = occupancyGrid
        marginGrid3D = GT.marginWithDepth(grid3D, desiredMarginDepthinMeters=0.1, pitchInMeters=grid_pitch)
        graph = GridRRTstar3D(start, goal, dgoal, dsearch, dcheaper, grid3D, marginGrid3D, max_iter)
        starttime = time.time()
        pbar = tqdm(total = max_iter)
        graph.makemap()
        while graph.iterations():
            graph.expand()
            pbar.update(1)
        endtime = time.time()
        pbar.close()
        print("Time elapsed: ", endtime - starttime)
        graph.makemap()

        # Convert path to simulation scale and frame
        path = np.array(graph.smoothpath).T
        path_refit = path*grid_pitch
        if savePath:
            np.save(os.path.join(relativepath,"smoothpath3D.npy"), path_refit)
    else: 
        path_refit = np.load(os.path.join(relativepath,"smoothpath3D.npy"))
else:
    # Generate sample path
    length = 400
    path_refit = np.zeros(length)
    for i, _ in range(length):
        path_refit[i] = [i*np.cos(i/length*np.pi*2), i*np.sin(i/length*np.pi*2), i/2]

# Generate waypoint
fine_waypoints = np.zeros(path_refit.shape)
for i in range(int(path_refit.shape[0]-1)): 
    fine_waypoints = np.linspace(path_refit[i], path_refit[i+1], int(control_duration*control_freq_hz))
fine_waypoints[-1, :] = path_refit[-1,:]
numWP = len(fine_waypoints)
wp_counter = 0

#### Initialize the simulation #############################
INIT_XYZS = np.array([initial_position for i in range(num_drones)])
INIT_RPYS = np.array([[0,0,0] for i in range(num_drones)])
AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

#### Create the environment 

env = CtrlAviary(drone_model=drone,
                    num_drones=num_drones,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    physics=physics,
                    neighbourhood_radius=10,
                    freq=simulation_freq_hz,
                    aggregate_phy_steps=AGGR_PHY_STEPS,
                    gui=gui,
                    record=record_video,
                    obstacles=obstacles,
                    user_debug_gui=user_debug_gui
                    )

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


for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
    
    p.loadURDF(obstacleToAdd) # load obstacle course
    
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
            wp_counter += 1 if wp_counter>numWP else 0
        yaw, pitch, roll = p.getEulerFromQuaternion(obs[str(j)]["state"][3:7])
        # p.resetDebugVisualizerCamera(0.001, 0, -roll*57.2958-30,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone POV
        # p.resetDebugVisualizerCamera(0.5, 0, -30,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone from behind
        # p.resetDebugVisualizerCamera(0.6, 0, -70,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone from above
        # p.resetDebugVisualizerCamera(0.01, -89.99,-89.99,[0,3,5]) # turn on to get view from above (static)
        p.resetDebugVisualizerCamera(3, -45,-45,[0,1,1]) # turn on to get view from above (static)

        #### Go to the next way point and loop #####################
        # for j in range(num_drones): 
        #     wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

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
    # if i%env.SIM_FREQ == 0:
    #     env.render()
    #     #### Print matrices with the images captured by each drone #
    #     if vision:
    #         for j in range(num_drones):
    #             print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
    #                   obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
    #                   obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
    #                   )

    #### Sync the simulation ###################################
    if gui:
        sync(i, START, env.TIMESTEP)

#### Close the environment #################################
env.close()

#### Save the simulation results ###########################
# logger.save()
# logger.save_as_csv("pid") # Optional CSV save

# #### Plot the simulation results ###########################
# if plot:
#     logger.plot()