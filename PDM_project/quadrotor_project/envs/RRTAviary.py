import os
import numpy as np
from datetime import datetime
import pybullet as p
import pybullet_data
import pkg_resources
from tqdm import tqdm
import time


from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

import quadrotor_project.planningAlgorithms.occupancyGridTools as GT
from quadrotor_project.planningAlgorithms.RRT import GridRRTstar2D, GridRRTstar3D

class RRTAviary(CtrlAviary):
    '''Drone environment derived from gym_pybullet's CtrlAviary with specific purpose of being used with our RRT* implementation'''

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles = True,
                 track = None,
                 tracking_dimension = 3,
                 load_path = True,
                 save_path = True,
                 user_debug_gui=True,
                 output_folder='results'):

        self.TRACK = track
        self.TRACKING_DIMENSION = tracking_dimension
        self.LOAD_PATH = load_path
        self.SAVE_PATH = save_path
        
        # Initialize RRT* settings
        self.grid_pitch = 0.05 # The size of the individual grid cells of the occupancy grid
        self.dgoal = 5 # Radius around the goal, if a node is generated within this radius and can be connected to the initial node through a number of edges, the goal is considered to be found
        self.dsearch = 10 # The radius within which the RRT* algorithms searches for the closest node after a new node has been generated
        self.dcheaper = 15 # The radius within which the algorithm searches for a node to reconnect to with a lower cost to the initial node

        ### Setting up track variables ###
        if self.TRACK !=0:
            # If track 0 is not selected, the obstacle to load is retrieved by self.getPath and self._addObstacles
            self.obstacletoadd = pkg_resources.resource_filename('quadrotor_project', 'assets/track{}.obj'.format(self.TRACK)) 

        '''Set positional, occupancygrid and RRT* variables for the given track.
        - initial_xyzs : the initial position of the quadrotor
        - initial_rpys : the initial attitude of the quadrotor
        - GOAL_XYZ : the position of the goal in 3D space
        - max_iter : the max amount of iterations the RRT* algorithm is allowed to run
        - track_time : the time the quadrotor has to follow the track's computed path
        - margindepth : the depth (in meters) of margins around obstacles that is to be generated in the occupancy grid
        '''
        if self.TRACK == 0:
            '''Track 0 is a demo track and does not use motionplanning, therefore it does not need the latter 4 settings'''
            pybullet_data.getDataPath() # Make sure that pybullet knows where to search for the urdf and associated obj file
            self.obstacletoadd = "samurai.obj"
            initial_xyzs = np.array([0,0,0.5], dtype = np.float64)
            initial_rpys = np.array([0,0,0], dtype=np.float64)
            
        elif self.TRACK == 1:
            initial_xyzs = np.array([0.5, 0.5, 0.8], dtype = np.float64)
            initial_rpys = np.array([0,0,0], dtype=np.float64)
            self.GOAL_XYZ = np.array([2, 5, 0.8], dtype = np.float64)
            self.max_iter = 8000
            self.track_time = 13
            self.margindepth = 0.2
        
        elif self.TRACK == 2:
            initial_xyzs = np.array([0.5, 0.5, 0.5], dtype = np.float64)
            initial_rpys = np.array([0,0,0], dtype=np.float64)
            self.GOAL_XYZ = np.array([6, 2, 2], dtype = np.float64)
            self.max_iter = 30000
            self.track_time = 10
            self.margindepth = 0.15

        elif self.TRACK == 3:
            initial_xyzs = np.array([0.5, 0.5, 0.5], dtype = np.float64)
            initial_rpys = np.array([0,0,0], dtype=np.float64)
            self.GOAL_XYZ = np.array([4, 0.5, 0.5], dtype = np.float64)
            self.max_iter = 130000
            self.track_time = 65
            self.margindepth = 0.25
        elif self.TRACK == 4:
            initial_xyzs = np.array([0.5, 0.5, 0.5], dtype = np.float64)
            initial_rpys = np.array([0,0,0], dtype=np.float64)
            self.GOAL_XYZ = np.array([4.5, 4.5, 0.5], dtype = np.float64)
            self.max_iter = 20000
            self.track_time = 5
            self.margindepth = 0.2
        else: 
            raise Exception("Track {} is not a valid track".format(self.TRACK))
                 
        # Leave the rest of the initiation to CtrlAviary and BaseAviary
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs.reshape(1,3),
                         initial_rpys=initial_rpys.reshape(1,3),
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder)

    ###############################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.
        Overridden function from BaseAviary
        Load obstacles from quadrotor_project/assets/ or pybullet's files, depending on the track

        """
        self.OBSTACLEID = p.loadURDF(self.obstacletoadd[:-3]+"urdf", physicsClientId=self.CLIENT)

    ###############################################################################

    def _startVideoRecording(self):
            """Starts the recording of a video output.
            Overridden function from BaseAviary
            The format of the video output is .mp4, if GUI is True, or .png, otherwise.
            The mp4 video is given a name containing the tracks and whether obstacles are displayed

            """
            if self.RECORD and self.GUI:
                recording_path = os.path.join(self.OUTPUT_FOLDER, "recordings")
                # os.makedirs(recording_path, exist_ok=True)
                self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                    fileName=os.path.join(recording_path, "output_track{}_obstacles_{}_".format(self.TRACK, self.OBSTACLES)+datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+".mp4"),
                                                    physicsClientId=self.CLIENT
                                                    )
            if self.RECORD and not self.GUI:
                self.FRAME_NUM = 0
                self.IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%d.%m.%Y_%H.%M.%S"), '')
                os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
        
    ###############################################################################

    def getPath(self):
        ''' Computes the path generated from an RRT* graph

            Returns
            -------
            ndarray[ndarray, ndarray, ndarray]
                arrays of equal length containg x, y and z coordinates respectively
            int
                tracking time
        
        '''
        if self.TRACK == 0:
            '''Make a demo spiral path'''
            length = 400
            path_refit = np.zeros((length, 3))
            for i in range(length):
                path_refit[i] = [i/800*np.cos(i/length*10*np.pi), i/800*np.sin(i/length*10*np.pi), i/600]+self.INIT_XYZS
        elif self.TRACK == 1:
            if not self.LOAD_PATH: # If we want to recompute a path based on the on the obj file, self.LOAD_PATH is set to False
                # Generating occupancy grid
                occupancyGrid, _ = GT.generateOccupancyGrid(self.obstacletoadd)
                
                # Applying RRT*
                start = (self.INIT_XYZS.reshape(3)[:2])/self.grid_pitch
                goal = (self.GOAL_XYZ.reshape(3)[:2])/self.grid_pitch
                grid2D = occupancyGrid[:,:,0]
                marginGrid2D = GT.marginWithDepth(grid2D, desiredMarginDepthinMeters=0.2, pitchInMeters=self.grid_pitch) # Apply a number of margins to the previously generated occupancy grid
                graph = GridRRTstar2D(start, goal, self.dgoal, self.dsearch, self.dcheaper, grid2D, marginGrid2D, self.max_iter) # Generate path using our 2D RRT* algorithm
                
                starttime = time.time()
                pbar = tqdm(total = self.max_iter) # initialize progress bar
                graph.makemap() # Plot the occupancy grid with initial position and goal position
                while graph.iterations(): # Iterate while the number of iterations is inferior to maxiter
                    graph.expand() # Generate new node and execute checks
                    pbar.update(1) # Advance progress bar
                endtime = time.time()
                pbar.close()
                print("Time elapsed: ", endtime - starttime)
                graph.makemap()
                # Convert path to simulation scale and frame
                path = np.array(graph.smoothpath).T
                path = np.append(path,self.INIT_XYZS.reshape(3)[2]/self.grid_pitch*np.ones((len(path),1)), axis = 1) # Append constant height to the path
                path_refit = path*self.grid_pitch
                if self.SAVE_PATH:
                    np.save(os.path.join(pkg_resources.resource_filename('quadrotor_project', 'assets/'),"track1.npy"), path_refit) # Save path as npy in assets folder
            else: 
                path_refit = np.load(pkg_resources.resource_filename('quadrotor_project', 'assets/track1.npy')) # Load npy  path file from assets folder
        
        else: # For all other tracks (not 0 or 1)
            if not self.LOAD_PATH: # If we want to recompute a path based on the on the obj file, self.LOAD_PATH is set to False
                # Generating occupancy grid
                occupancyGrid, _ = GT.generateOccupancyGrid(self.obstacletoadd)
                
                # Applying RRT*
                start = (self.INIT_XYZS.reshape(3))/self.grid_pitch
                goal = (self.GOAL_XYZ.reshape(3))/self.grid_pitch
                grid3D = occupancyGrid
                marginGrid3D = GT.marginWithDepth(grid3D, desiredMarginDepthinMeters=self.margindepth, pitchInMeters=self.grid_pitch) # Add a number of margin layers to the occupancy grid
                graph = GridRRTstar3D(start, goal, self.dgoal, self.dsearch, self.dcheaper, grid3D, marginGrid3D, self.max_iter) # Feed the occupancy grid into our 3D RRT* algorithm
                starttime = time.time()
                pbar = tqdm(total = self.max_iter) # Initialize progress bar
                graph.makemap() # Plot the occupancy grid with initial position and goal position
                while graph.iterations(): # Iterate while the number of iterations is inferior to maxiter
                    graph.expand() # Generate new node and execute checks
                    pbar.update(1) # Advance progress bar
                endtime = time.time()
                pbar.close()
                print("Time elapsed: ", endtime - starttime)
                graph.makemap()

                # Convert path to simulation scale and frame
                path = np.array(graph.smoothpath).T
                path_refit = path*self.grid_pitch
                if self.SAVE_PATH:
                    np.save(os.path.join(pkg_resources.resource_filename('quadrotor_project', 'assets/'),"track{}.npy".format(self.TRACK)), path_refit) # Save path as npy in assets folder
            else: 
                path_refit = np.load(pkg_resources.resource_filename('quadrotor_project', 'assets/track{}.npy'.format(self.TRACK))) # Load npy  path file from assets folder
            
        return path_refit, self.track_time

    ###############################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, bool]
            Dummy value.

        """
        return {"is_yeet": True} #### GenZ communication