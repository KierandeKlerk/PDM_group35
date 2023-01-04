import pybullet as p
import time
import pybullet_data
import numpy as np
import pkg_resources

from quadrotor_project.envs.PointMassAviary import PointMassAviary
from quadrotor_project.control.pointmassPID import pointmassPID

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync


### Settings ###
drone = DroneModel("cf2p")
physics = Physics("dyn")
gui = True
record = True # If set to true, will record an mp4; somehow 4 times faster than simulation time
aggregate = True
obstacles = True
simulation_freq_hz = 240
control_freq_hz = 48
duration_sec = 12
periodic = True # Controls whether the drone starts over when reaching the last target waypoint

### Initial pose ###
INIT_XYZ = np.array([0,0,0.8], dtype=float)
INIT_RPY = np.array([0,0,0], dtype=float)
AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1

### Target position calculation (to be replaced with data from RRT) ###
R = 0.6

TRACK_TIME = 4
NUM_WAY_POINTS = TRACK_TIME*control_freq_hz
print("Number of waypoints {}".format(NUM_WAY_POINTS))
TARGET_POS = np.zeros((NUM_WAY_POINTS,3))
for i in range(NUM_WAY_POINTS):
    TARGET_POS[i, :] = [R*np.cos(i/NUM_WAY_POINTS*np.pi*2)-R, R*np.sin(i/NUM_WAY_POINTS*2*np.pi), 0.8]
wp_counter = 0

### Initiate environment ###
env = PointMassAviary(drone_model=drone,
                      initial_xyz=INIT_XYZ,
                      initial_rpy=INIT_RPY,
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
            if periodic:
                wp_counter = wp_counter + 1 if wp_counter < NUM_WAY_POINTS-1 else 0
            else:
                wp_counter += 1 if wp_counter < NUM_WAY_POINTS-1 else 0
        ### Printout data ###
        if i%env.SIM_FREQ == 0:
            env.render()
        
        ### Sync the simulation ###
        if gui:
            sync(i, START, env.TIMESTEP)
env.close()