import pybullet as p
import time
import pybullet_data
import numpy as np
import pkg_resources
from tqdm import tqdm
import os

from quadrotor_project.envs.PointMassAviary import PointMassAviary
from quadrotor_project.control.pointmassPID import pointmassPID
from quadrotor_project.planningAlgorithms.occupancyGridTools import generateOccupancyGrid, marginWithDepth
from quadrotor_project.planningAlgorithms.RRT import GridRRTstar2D

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync


### Settings ###
drone = DroneModel("cf2p")
physics = Physics("dyn")
gui = True
record = False # If set to true, will record an mp4; somehow 4 times faster than simulation time
aggregate = True
obstacles = True
simulation_freq_hz = 240
control_freq_hz = 48
duration_sec = 20           
# Obstacle and location settings
obstacleFile = pkg_resources.resource_filename("quadrotor_project", "assets/drone_parcours.obj")
gridPitch = 0.05
height = 0.8 # in meters 
start_xy = np.array([0.5, 0.5]) # in meters
goal_xy = np.array([2, 5]) # in _meters
dgoal = 5
dsearch = 10
dcheaper = 15
max_iter = 6000
track_time = 18
loadPath = True
savePath = False
relativepath = "PDM/Project/PDM_group35/PDM_project/"

### Initial pose ###

AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

### Target position calculation through RRT* ###

if not loadPath:
    # Generating occupancy grid
    occupancyGrid, offsets = generateOccupancyGrid(obstacleFile)
    print(offsets)
    # Applying RRT*
    start = (start_xy-offsets[:2])/gridPitch
    goal = (goal_xy-offsets[:2])/gridPitch
    grid2D = occupancyGrid[:,:,0]
    marginGrid2D = marginWithDepth(grid2D, desiredMarginDepthinMeters=0.2, pitchInMeters=gridPitch)
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
    path = np.append(path,height/gridPitch*np.ones((len(path),1)), axis = 1)
    path_refit = path*gridPitch+offsets
    if savePath:
        np.save(os.path.join(relativepath,"track1.npy"), path_refit)
else: 
    path_refit = np.load(os.path.join(relativepath,"track1.npy"))

NUM_WAY_POINTS = int(track_time*control_freq_hz)
TARGET_POS = np.zeros((NUM_WAY_POINTS,3))
ind = np.round(np.linspace(0, len(path_refit) - 1, NUM_WAY_POINTS)).astype(int)
TARGET_POS = path_refit[ind,:]
wp_counter = 0

### Initiate environment ###
env = PointMassAviary(drone_model=drone,
                      initial_xyz=np.append(start_xy, [height]),
                      initial_rpy=np.array([0,0,0], dtype=float),
                      physics=physics,
                      freq=simulation_freq_hz,
                      aggregate_phy_steps=AGGR_PHY_STEPS,
                      gui=gui,
                      record=record,
                      obstacles=obstacles)

### Initiate PID controller ###
ctrl = pointmassPID(drone_model=drone)

### Run simulation ###
CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
action = np.zeros(3)
START = time.time()
done = False

for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS): 
    ### Continue simulation if the goal hasn't been reached yet ###
    
    if not done:
        
        ### Step the simulation ###
        state, done, _ = env.step(action)

        ### Compute control at the desired  frequency
        if i%CTRL_EVERY_N_STEPS == 0:

            ### Compute control for the current waypoint ###
            action = ctrl.computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                  state=state,
                                                  target_pos=TARGET_POS[wp_counter])

            ### Select next waypoint ###
            wp_counter += 1 if wp_counter < NUM_WAY_POINTS-1 else 0
        ### Printout data ###
        if i%env.SIM_FREQ == 0:
            env.render()
        
        ### Sync the simulation ###
        if gui:
            sync(i, START, env.TIMESTEP)
env.close()