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
#sys.path.append('../gym-pybullet-drones')

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

import pybullet as p
import time
import pybullet_data
import pkg_resources
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False

DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
# DEFAULT_DURATION_SEC = 18
DEFAULT_DURATION_SEC = 7
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

### From RRT
x = np.array([10.14588987, 10.86982641, 11.66107733, 12.51450297, 13.42496369,
       14.38731984, 15.39643178, 16.44715986, 17.53436444, 18.65290586,
       19.79764448, 20.96344066, 22.14515475, 23.33764709, 24.53577806,
       25.73440799, 26.92839724, 28.11260617, 29.28189513, 30.43112448,
       31.55515456, 32.64884573, 33.70705834, 34.72465275, 35.69648931,
       36.61742838, 37.4823303 , 38.28605544, 39.02346414, 39.69054719,
       40.28911167, 40.82282996, 41.29537531, 41.71042095, 42.07164015,
       42.38270614, 42.64729218, 42.86907151, 43.05171738, 43.19890303,
       43.31430171, 43.40158667, 43.46443115, 43.5065084 , 43.53149168,
       43.54305422, 43.54486927, 43.5405901 , 43.53262031, 43.52139686,
       43.50718397, 43.4902459 , 43.47084687, 43.44925114, 43.42572294,
       43.40052652, 43.37392612, 43.34618597, 43.31757032, 43.28834342,
       43.25876949, 43.22911279, 43.19963755, 43.17059838, 43.14109993,
       43.10802168, 43.06798861, 43.01762571, 42.95355795, 42.87241031,
       42.77080778, 42.64537534, 42.49273797, 42.30952064, 42.09234835,
       41.83784607, 41.54263878, 41.20335146, 40.8166091 , 40.37903667,
       39.88725916, 39.33961945, 38.74133328, 38.09933491, 37.42055859,
       36.71193857, 35.98040911, 35.23290447, 34.4763589 , 33.71770665,
       32.96388198, 32.22100011, 31.49018411, 30.77067638, 30.06171585,
       29.36254149, 28.67239224, 27.99050706, 27.3161249 , 26.64848471,
       25.98682544, 25.33038604, 24.67840547, 24.03012267, 23.3847766 ,
       22.74160621, 22.09985045, 21.45874828, 20.81753863, 20.17546047,
       19.53175275, 18.88565442, 18.23640442, 17.58324172, 16.92540526,
       16.262134  , 15.59266688, 14.91624286, 14.23210089, 13.53947992,
       12.83811707, 12.13535191, 11.44445311, 10.77884635, 10.15195732,
        9.57721173,  9.06803524,  8.63785357,  8.29991597,  8.05829295,
        7.90358257,  7.82530791,  7.81299207,  7.85615811,  7.94432913,
        8.06702821,  8.21377842,  8.37410285,  8.53752458,  8.6935667 ,
        8.83175229,  8.94160567,  9.01563942,  9.05576128,  9.06572069,
        9.04926711,  9.01014996,  8.9521187 ,  8.87892277,  8.79431161,
        8.70203467,  8.60584139,  8.50948121,  8.41670357,  8.33125793,
        8.25689372,  8.19736038,  8.15640737,  8.13778411,  8.14524007,
        8.18252468,  8.25338738,  8.36157761,  8.51084483,  8.70493848,
        8.94760799,  9.24260281,  9.59367239, 10.00456616, 10.47903358,
       11.02055797, 11.62855172, 12.29924715, 13.02879207, 13.81333427,
       14.64902156, 15.53200174, 16.45842262, 17.424432  , 18.42617768,
       19.45980746, 20.52146915, 21.60731055, 22.71347946, 23.83612368,
       24.97139103, 26.1154293 , 27.26438629, 28.4144098 , 29.56164765,
       30.70224763, 31.83235755, 32.9481252 , 34.0456984 , 35.12122494,
       36.17085263, 37.19072927, 38.17700267, 39.12582062, 40.03333093])
       
y = np.array([  9.96676636,  10.36275187,  10.70997496,  11.01259844,
    11.27478512,  11.5006978 ,  11.6944993 ,  11.86035241,
    12.00241995,  12.12486472,  12.23184953,  12.32753719,
    12.4160905 ,  12.50167227,  12.58844531,  12.68057243,
    12.78221642,  12.89754011,  13.03070629,  13.18587777,
    13.36721736,  13.57888787,  13.8250521 ,  14.10987286,
    14.43751296,  14.81213521,  15.2379024 ,  15.71897735,
    16.25952287,  16.86265063,  17.52606407,  18.24573216,
    19.0176231 ,  19.83770505,  20.7019462 ,  21.60631472,
    22.5467788 ,  23.51930662,  24.51986635,  25.54442618,
    26.58895428,  27.64941883,  28.72178802,  29.80203002,
    30.88611302,  31.97000519,  33.0496747 ,  34.12112721,
    35.18271048,  36.23645836,  37.28472835,  38.32987796,
    39.37426468,  40.42024604,  41.47017953,  42.52642266,
    43.59133295,  44.66726789,  45.75658499,  46.86164175,
    47.9847957 ,  49.12840432,  50.29482514,  51.48638565,
    52.70183227,  53.93298206,  55.17085954,  56.40648921,
    57.63089559,  58.8351032 ,  60.01013654,  61.14702013,
    62.23677848,  63.2704361 ,  64.23901751,  65.13354722,
    65.94504974,  66.66454958,  67.28307126,  67.79163929,
    68.18127818,  68.44534281,  68.58651113,  68.60979233,
    68.52019554,  68.32272994,  68.02240469,  67.62422894,
    67.13321185,  66.55436259,  65.89269032,  65.15347711,
    64.34366852,  63.47083681,  62.54255535,  61.56639755,
    60.54993678,  59.50074643,  58.4263999 ,  57.33447057,
    56.23253183,  55.12815708,  54.02891969,  52.94239306,
    51.87615057,  50.83776562,  49.8348116 ,  48.87486188,
    47.96548987,  47.11426895,  46.32877251,  45.61657393,
    44.98524661,  44.44236394,  43.9954993 ,  43.65222608,
    43.42011767,  43.30674746,  43.31968885,  43.4665152 ,
    43.75400223,  44.17675226,  44.7198737 ,  45.36822349,
    46.1066586 ,  46.92003598,  47.7932126 ,  48.71104541,
    49.65849849,  50.62610901,  51.61259417,  52.61732389,
    53.63966809,  54.6789967 ,  55.73467962,  56.80608678,
    57.89258809,  58.99355348,  60.10835285,  61.23635614,
    62.37693325,  63.52945399,  64.69300733,  65.86579932,
    67.04586295,  68.23123117,  69.41993696,  70.61001329,
    71.79949314,  72.98640948,  74.16879528,  75.34468351,
    76.51210715,  77.66909916,  78.81369253,  79.94392022,
    81.05781521,  82.15341047,  83.22873897,  84.28183368,
    85.31072758,  86.31345364,  87.28804483,  88.23253412,
    89.14495449,  90.02333891,  90.86572035,  91.67013179,
    92.43460619,  93.15717653,  93.83599145,  94.47096887,
    95.06340885,  95.61464818,  96.12602364,  96.59887203,
    97.03453013,  97.43433473,  97.79962261,  98.13173056,
    98.43199537,  98.70175383,  98.94234272,  99.15509883,
    99.34135895,  99.50245987,  99.63973836,  99.75453123,
    99.84817525,  99.92200722,  99.97736391, 100.01558212,
    100.03799864, 100.04595025, 100.04077374, 100.0238059 ,
    99.99638351,  99.95984336,  99.91552223,  99.86475693])
z = np.ones(y.shape)*0.8
print(x.shape, y.shape, z.shape)
x = x * 0.05 
y = y * 0.05
waypoints = np.vstack((np.vstack((x.T,y.T)), z.T)).T
print(waypoints[195:])
print(waypoints.shape)
WAYPOINTS_INPUT = waypoints
WAYPOINTS_INPUT = np.array([[2,2,2.5], [2,2,2.5],[2,2,2.5]])
# WAYPOINTS_INPUT = np.array([[0,0,0.8], [0,0,0.8], [1.6, 0, 0.8], [1.6, 3.2, 0.8], [0.8, 3.2, 0.8], [0.8, 1, 0.8], [-0.1, 1, 0.8], [-0.1, 4.6, 0.8], [1.6, 4.6, 0.8], [1.6, 4.6, 0.8]])
# WAYPOINTS_INPUT = np.array([[0,0,0.8], [1.6, 0, 0.8], [1.6, 3.2, 0.8], [0.8, 3.2, 0.8], [0.8, 1, 0.8], [-0.1, 1, 0.8], [-0.1, 4.6, 0.8], [1.6, 4.6, 0.8]])
obstacle_to_add = pkg_resources.resource_filename('quadrotor_project', 'assets/Hole_obstacle_course.urdf')
print(WAYPOINTS_INPUT.shape)

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        vision=DEFAULT_VISION,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        waypoints = WAYPOINTS_INPUT
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([WAYPOINTS_INPUT[0] for i in range(num_drones)])
    INIT_RPYS = np.array([[0,0,0] for i in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1
    


    #### Initialize a circular trajectory ######################
    # PERIOD = 10
    # NUM_WP = control_freq_hz*PERIOD
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     TARGET_POS[i, :] = 1, 0, 0.8
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

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
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    elif drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(num_drones)}
    START = time.time()


    new_finer_waypoints = np.array([waypoints[0]]) # add start point to trajectory
    num_time_per_waypoint = duration_sec/ (len(waypoints)-1) # get time it takes to complete one trajectory inside of the full one

    ### Make the trajectory the size of the simulation, so that every i in simulation has a different point in the trajectory ###
    for i in range(int(waypoints.shape[0]-1)): 
        waypoints_finer_traj = np.linspace(waypoints[i], waypoints[i+1], int(num_time_per_waypoint*env.SIM_FREQ))
        new_finer_waypoints =  np.vstack((new_finer_waypoints, waypoints_finer_traj[1:]))

    for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
        
        p.loadURDF(obstacle_to_add) # load obstacle course
        
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        
        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:
            #### Compute control for the current way point #############
            for j in range(num_drones):
                target_now = new_finer_waypoints[i]
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos=target_now,
                                                                       target_rpy=INIT_RPYS[j, :]
                                                                       )
            yaw, pitch, roll = p.getEulerFromQuaternion(obs[str(j)]["state"][3:7])
            # p.resetDebugVisualizerCamera(0.001, 0, -roll*57.2958-30,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone POV
            # p.resetDebugVisualizerCamera(0.5, 0, -30,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone from behind
            # p.resetDebugVisualizerCamera(1, 89, 0,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone from behind test
            # # p.resetDebugVisualizerCamera(0.6, 0, -70,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone from above
            # p.resetDebugVisualizerCamera(0.01, -89.99,-89.99,[0,3,5]) # turn on to get view from above (static)
            # p.resetDebugVisualizerCamera(3, -45,-45,[0,1,1]) # turn on to get view from above (static)

            #### Go to the next way point and loop #####################
            # for j in range(num_drones): 
            #     wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

            # print(f'Time right now: {i/env.SIM_FREQ}')

        #### Log the simulation ####################################
        # for j in range(num_drones):
        #     logger.log(drone=j,
        #                timestamp=i/env.SIM_FREQ,
        #                state=obs[str(j)]["state"],
        #                control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
        #                # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
        #                )

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

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=DEFAULT_VISION,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))