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
x = np.array([  9.99969769,  10.51403988,  11.03594191,  11.56488042,
        12.100332  ,  12.64177327,  13.18868085,  13.74053134,
        14.29680136,  14.85696752,  15.42050644,  15.98689473,
        16.555609  ,  17.12612586,  17.69792193,  18.27047381,
        18.84325813,  19.4157515 ,  19.98743052,  20.55777182,
        21.12625199,  21.69234767,  22.25553545,  22.81529196,
        23.3710938 ,  23.92241759,  24.46873994,  25.00953747,
        25.54428678,  26.07246449,  26.59354721,  27.10701156,
        27.61245285,  28.11059906,  28.60280655,  29.09043832,
        29.5748574 ,  30.05742684,  30.53950964,  31.02246884,
        31.50766747,  31.99646856,  32.49023513,  32.9903302 ,
        33.49811681,  34.01495798,  34.54221675,  35.08125613,
        35.63343916,  36.20012886,  36.78268826,  37.38248039,
        38.00086827,  38.63921494,  39.29888341,  39.98123673,
        40.6876379 ,  41.41944997,  42.17793067,  42.96068214,
        43.76079647,  44.5710872 ,  45.38436787,  46.193452  ,
        46.99115313,  47.7702848 ,  48.52377946,  49.24885818,
        49.94812719,  50.62453451,  51.28102815,  51.92055615,
        52.54606651,  53.16050725,  53.7668264 ,  54.36797198,
        54.96689199,  55.56653447,  56.16984736,  56.7790377 ,
        57.39375152,  58.01308182,  58.63612165,  59.26196401,
        59.88970193,  60.51842844,  61.14723655,  61.7752193 ,
        62.4014697 ,  63.02508077,  63.64514555,  64.26075704,
        64.87100828,  65.47499229,  66.07180209,  66.66053071,
        67.24027116,  67.81012318,  68.36989369,  68.92071027,
        69.46384508,  70.00057029,  70.53215806,  71.05988058,
        71.58501002,  72.10881853,  72.63257829,  73.15756148,
        73.68504026,  74.2162868 ,  74.75257328,  75.29517186,
        75.84535471,  76.404394  ,  76.97356191,  77.5541306 ,
        78.14737225,  78.75455902,  79.37696308,  80.01585661,
        80.67251178,  81.34820075,  82.04419569,  82.76176878,
        83.50219155,  84.26501886,  85.04435662,  85.83323204,
        86.62467235,  87.41170479,  88.18735657,  88.94465494,
        89.67691126,  90.38204997,  91.06174106,  91.71776279,
        92.35189342,  92.96591122,  93.56159443,  94.14072133,
        94.70507017,  95.25641923,  95.79654675,  96.327231  ,
        96.85025025,  97.36738275,  97.88040677,  98.39110056,
        98.90124239,  99.41261052,  99.92698322, 100.44613873,
       100.97185534, 101.50591128, 102.05008484, 102.60615427,
       103.17589783, 103.76109378, 104.36330816, 104.98196476,
       105.61524539, 106.26131662, 106.91834502, 107.58449717,
       108.25793964, 108.93683901, 109.61936186, 110.30367475,
       110.98794426, 111.67033696, 112.34901944, 113.02215826,
       113.68792   , 114.34447124, 114.98997854, 115.62260848,
       116.24052765, 116.8419026 , 117.42489993, 117.98768619,
       118.52842797, 119.04529184, 119.53644437, 120.00005214])
y = np.array([ 9.93975708, 10.45474044, 10.96637552, 11.47494576, 11.98073458,
       12.48402542, 12.98510171, 13.48424688, 13.98174436, 14.47787758,
       14.97292997, 15.46718497, 15.96092601, 16.45443651, 16.94799991,
       17.44189964, 17.93641913, 18.43184182, 18.92845112, 19.42653048,
       19.92636333, 20.4282331 , 20.93242321, 21.4392171 , 21.94889821,
       22.46174995, 22.97805577, 23.4980991 , 24.02216336, 24.55053199,
       25.08348842, 25.62131607, 26.16419278, 26.71128853, 27.26121425,
       27.8125749 , 28.36397546, 28.91402089, 29.46131616, 30.00446624,
       30.5420761 , 31.07275069, 31.595095  , 32.10771398, 32.60921261,
       33.09819585, 33.57326867, 34.03303603, 34.47610291, 34.90107427,
       35.30655509, 35.69115032, 36.05346493, 36.39210389, 36.70567218,
       36.99277475, 37.25201658, 37.48200262, 37.68131549, 37.84776089,
       37.9781861 , 38.06937922, 38.1181283 , 38.12122144, 38.07544671,
       37.97759218, 37.8245499 , 37.61696098, 37.36017418, 37.05983709,
       36.72159726, 36.35110225, 35.95399964, 35.53593699, 35.10256186,
       34.65952182, 34.21246444, 33.76703728, 33.32888776, 32.9022388 ,
       32.48638916, 32.07957439, 31.68003005, 31.28599166, 30.89569478,
       30.50737496, 30.11926774, 29.72960866, 29.33663328, 28.93857713,
       28.53367576, 28.12016472, 27.69627955, 27.2602558 , 26.81032902,
       26.34473474, 25.86170852, 25.35949763, 24.83758576, 24.29776549,
       23.74208225, 23.17258144, 22.59130849, 22.00030879, 21.40162777,
       20.79731084, 20.1894034 , 19.57995088, 18.97099869, 18.36459224,
       17.76277694, 17.16759821, 16.58110145, 16.00533209, 15.44233553,
       14.89415719, 14.36284248, 13.85043682, 13.35898561, 12.89053427,
       12.44712822, 12.03081286, 11.64363361, 11.28763588, 10.96486525,
       10.67780207, 10.43030675, 10.22651291, 10.07055415,  9.96656408,
        9.91867632,  9.93102446, 10.00750114, 10.14808685, 10.34958565,
       10.60870974, 10.92217137, 11.28668274, 11.69895609, 12.15570363,
       12.6536376 , 13.18947021, 13.75991369, 14.36168025, 14.99148213,
       15.64603155, 16.32204072, 17.01622188, 17.72528725, 18.44594904,
       19.17491949, 19.90891081, 20.64463524, 21.37880498, 22.10813228,
       22.82932934, 23.53910839, 24.23418166, 24.91154365, 25.57103821,
       26.21416117, 26.84242857, 27.4573565 , 28.06046102, 28.65325819,
       29.23726408, 29.81399476, 30.38496629, 30.95169473, 31.51569617,
       32.07848665, 32.64158225, 33.20649903, 33.77475306, 34.34786041,
       34.92733713, 35.51469931, 36.111463  , 36.71914427, 37.33925918,
       37.97332381, 38.62285421, 39.28936646, 39.97437662])
z = np.array([ 9.96319132, 10.86739768, 11.73235639, 12.56036836, 13.35373451,
       14.11475576, 14.84573302, 15.5489672 , 16.22675923, 16.88141001,
       17.51522047, 18.13049152, 18.72952408, 19.31461906, 19.88807737,
       20.45219994, 21.00928768, 21.5616415 , 22.11156232, 22.66135106,
       23.21330863, 23.76973594, 24.33293392, 24.90520348, 25.48884554,
       26.086161  , 26.69945079, 27.33101583, 27.98315702, 28.65817529,
       29.35837154, 30.0860467 , 30.84312409, 31.62792359, 32.43676625,
       33.26595179, 34.11177991, 34.97055034, 35.83856279, 36.71211698,
       37.58751263, 38.46104945, 39.32902715, 40.18774546, 41.03350409,
       41.86260276, 42.67134119, 43.45601908, 44.21293616, 44.93839214,
       45.62868674, 46.28011968, 46.88899067, 47.45159943, 47.96424568,
       48.42322913, 48.82484949, 49.16540649, 49.44119197, 49.64822445,
       49.78218523, 49.83873481, 49.81353366, 49.70224227, 49.50052111,
       49.20403067, 48.80861809, 48.31686193, 47.73979334, 47.08897995,
       46.37598939, 45.61238927, 44.80974722, 43.97963086, 43.13360781,
       42.2832457 , 41.44011214, 40.61577477, 39.82180091, 39.06677551,
       38.34897435, 37.66444722, 37.00924393, 36.37941427, 35.77100803,
       35.18007502, 34.60266502, 34.03482785, 33.47261328, 32.91207113,
       32.34925118, 31.78020323, 31.20097708, 30.60762253, 29.99618938,
       29.36272741, 28.70328643, 28.01394099, 27.29337547, 26.54514789,
       25.77334994, 24.9820733 , 24.17540964, 23.35745065, 22.532288  ,
       21.70401338, 20.87671847, 20.05449493, 19.24143447, 18.44162875,
       17.65916945, 16.89814826, 16.16265685, 15.4567869 , 14.7846301 ,
       14.15027813, 13.55782265, 13.01135536, 12.51496794, 12.07275206,
       11.6887994 , 11.36720164, 11.11205047, 10.92743755, 10.8174539 ,
       10.78436433, 10.82463697, 10.93359251, 11.1065516 , 11.33883491,
       11.6257631 , 11.96265686, 12.34489462, 12.76879312, 13.23143092,
       13.72990857, 14.26132665, 14.82278573, 15.41138639, 16.02422919,
       16.6584147 , 17.3110435 , 17.97921616, 18.66003324, 19.35059533,
       20.04800298, 20.74935677, 21.45175728, 22.15230507, 22.84810071,
       23.53624477, 24.21383784, 24.87798047, 25.52577323, 26.15431671,
       26.76071147, 27.34205808, 27.89545711, 28.41836816, 28.91187486,
       29.37916195, 29.82343992, 30.24791926, 30.65581045, 31.05032398,
       31.43467035, 31.81206003, 32.18570353, 32.55881133, 32.93459392,
       33.31626179, 33.70702543, 34.11009532, 34.52868196, 34.96599584,
       35.42524744, 35.90964725, 36.42240577, 36.96673348, 37.54584087,
       38.16293843, 38.82123665, 39.52394602, 40.27427703])
print(x.shape, y.shape, z.shape)
x = x * 0.05 
y = y * 0.05
z = z * 0.05 
waypoints = np.vstack((np.vstack((x.T,y.T)), z.T)).T
print(waypoints[195:])
print(waypoints.shape)
WAYPOINTS_INPUT = waypoints
# WAYPOINTS_INPUT = np.array([[2,2,2.5], [2,2,2.5],[2,2,2.5]])
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
            p.resetDebugVisualizerCamera(0.5, 0, -30,obs[str(j)]["state"][:3]- [0,0,0]) # turn on to track the drone from behind
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