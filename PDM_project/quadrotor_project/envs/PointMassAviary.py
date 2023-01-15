######################################################################################
# This file was adapted from gym-pybullet-drone's different aviary files (namely BaseAviary and others in lesser degree)
######################################################################################

import os
import time
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from gym_pybullet_drones.utils.enums import Physics, DroneModel


class PointMassAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    metadata = {'render.modes': ['human']}
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2P,
                 initial_xyz=None,
                 initial_rpy=None,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 dynamics_attributes=False,
                 output_folder='results'
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyz: ndarray | None, optional
            (3,)-shaped array containing the initial XYZ position of the drone.
        initial_rpy: ndarray | None, optional
            (3,)-shaped array containing the initial orientation of the drone (in radians).
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to 'PointMassAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        dynamics_attributes : bool, optional
            Whether to allocate the attributes needed by subclasses accepting thrust and torques inputs.
        output_folder: str
            Where to store log and recording outputs
        """
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        #### Parameters ############################################
        
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        print("[INFO] PointMassAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        
        #### Create attributes for dynamics control inputs #########
        self.DYNAMICS_ATTR = dynamics_attributes
        if self.DYNAMICS_ATTR:
            if self.DRONE_MODEL == DroneModel.CF2X:
                self.A = np.array([ [1, 1, 1, 1], [1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2)], [-1, 1, -1, 1] ])
            elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
                self.A = np.array([ [1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1] ])
            self.INV_A = np.linalg.inv(self.A)
            self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=4.3,
                                         cameraYaw=-90,
                                         cameraPitch=-50,
                                         cameraTargetPosition=[1, 2.5, 0.8],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.SIM_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        if initial_xyz is None:
            self.INIT_XYZ = [0,0,1]
        elif np.array(initial_xyz).shape == (3,):
            self.INIT_XYZ = initial_xyz
        else:
            print("[ERROR] invalid initial_xyz in PointMassAviary.__init__(), try initial_xyz.reshape(3,1)")
        if initial_rpy is None:
            self.INIT_RPY = np.zeros(3)
        elif np.array(initial_rpy).shape == (3,):
            self.INIT_RPY = initial_rpy
        else:
            print("[ERROR] invalid initial_rpy in PointMassAviary.__init__(), try initial_rpy.reshape(3,1)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
    
    ################################################################################

    def reset(self):
        """Resets the environment.

        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        return -1 # Dummy value
    
    ################################################################################

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray 
            (3,), The input action for the drone, in the form of values for the commanded thrust, roll and pitch

        Returns
        -------
        ndarray
            (16,), This step's state.
        dict[..]
            Additional information as a dictionary.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            self.FRAME_NUM += 1
        #### Read the GUI's input parameters #######################
        #### Save, preprocess, and clip the action to the max. RPM #
        self._saveLastAction(action)
        clipped_action = np.reshape(self._preprocessAction(action), (3,1))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.AGGR_PHY_STEPS):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.AGGR_PHY_STEPS > 1 :
                self._updateAndStoreKinematicInformation()
            #### Step the simulation ##
            self._dynamics(clipped_action)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        state = self._getDroneStateVector()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)

        return state, info  
    
    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)
    
    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        int.

        """
        return self.DRONE_ID
    
    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1
        self.Y_AX = -1
        self.Z_AX = -1
        self.GUI_INPUT_TEXT = -1
        self.last_input_switch = 0
        self.last_action = -1*np.ones((3,1))
        self.last_clipped_action = np.zeros((3,1))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinematic information ##########
        self.pos = np.zeros((3,1))
        self.quat = np.zeros((4,1))
        self.rpy = np.zeros((3,1))
        self.vel = np.zeros((3,1))
        self.ang_v = np.zeros((3,1))
        self.rpy_rates = np.zeros((3,1))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.DRONE_ID =p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.INIT_XYZ,
                                              p.getQuaternionFromEuler(self.INIT_RPY),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              )
        if self.OBSTACLES:
            self._addObstacles()
    
    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        pos, quat = p.getBasePositionAndOrientation(self.DRONE_ID, physicsClientId=self.CLIENT)
        self.pos = np.array(pos)
        self.quat = np.array(quat)
        rpy = p.getEulerFromQuaternion(self.quat)
        self.rpy = np.array(rpy)
        vel, ang_v = p.getBaseVelocity(self.DRONE_ID, physicsClientId=self.CLIENT)
        self.vel = np.array(vel)
        self.ang_v = np.array(ang_v)
    
    ################################################################################

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.

        """
        if self.RECORD and self.GUI:
            recording_path = os.path.join(self.OUTPUT_FOLDER, "recordings")
            # os.makedirs(recording_path, exist_ok=True)
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=os.path.join(recording_path, "output"+datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+".mp4"),
                                                physicsClientId=self.CLIENT
                                                )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%d.%m.%Y_%H.%M.%S"), '')
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
    
    ################################################################################

    def _getDroneStateVector(self):
        """Returns the state vector of the drone.

        Returns
        -------
        ndarray 
            (16,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        
        state = np.hstack((self.pos, self.quat, self.rpy,self.vel, self.last_clipped_action.reshape((-1,))))
        print(state.shape)
        return state.reshape(-1,)
    
    ################################################################################

    def _dynamics(self,
                  action
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        ndarray
            (3,), commanded thrust, roll and pitch

        """
        #### Current state #########################################
        pos = self.pos
        vel = self.vel
        rotation = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        #### Compute kinematics ############################
        thrust = action[0]
        rpy = np.append(action[1:],[0])
        thrust_world_frame = thrust*np.dot(rotation, np.array([0,0,1]))
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        no_pybullet_dyn_accs = force_world_frame / self.M
        vel = vel + self.TIMESTEP * no_pybullet_dyn_accs
        pos = pos + self.TIMESTEP * vel

        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_ID,
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.CLIENT
                                          )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_ID,
                            vel,
                            [-1, -1, -1], 
                            physicsClientId=self.CLIENT
                            )
        #### Store the roll, pitch, yaw and respective rates for the next step ####
        self.rpy = rpy
        self.rpy_rates = [0., 0., 0.]
    
    ################################################################################

    def _saveLastAction(self,
                        action):
        """Stores the most recent action into attribute `self.last_action`.

        The last action can be used to compute aerodynamic effects.
        The method disambiguates between array and dict inputs 
        (for single or multi-agent aviaries, respectively).

        Parameters
        ----------
        action : ndarray
            (3)-shaped array of floats containing the current thrust and orientation input.

        """
        if isinstance(action, np.ndarray):
            self.last_action = action
        else: 
            res_action = np.resize(action, (3,1)) # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
            self.last_action = np.reshape(res_action, (3,1))
    
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from the assets folder of our quadrotor_project package

        """
        p.loadURDF(pkg_resources.resource_filename('quadrotor_project', 'assets/track1.urdf'),
                   physicsClientId=self.CLIENT
                   )
    
    ################################################################################
    
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3

    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

           Returns
           -------
            Box(4,)
            Contains: commanded thrust, commanded roll and pitch in world frame
        
        """

        #### Action vector ######## Thrust              Roll        Pitch
        act_lower_bound = np.array([0.,                -np.pi/4,   -np.pi/4])
        act_upper_bound = np.array([self.MAX_THRUST,    np.pi/4,    np.pi/4])
        return spaces.Box(low=act_lower_bound,
                          high=act_upper_bound,
                          dtype=np.float64)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

           Returns
           -------
            Box(16,)
            State vector contaning: position in world frame; quaternion orientation in world frame; roll, pitch and yaw in world frame; velocity and body frame, velocity in world frame

        """
        #### Observation vector ### X        Y        Z,      Q1   Q2   Q3   Q4,  R       P       Y,      VX       VY       VZ,      WX       WY       WZ       
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])
        return spaces.Box(low=obs_lower_bound,
                          high=obs_upper_bound,
                          dtype=np.float64)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into valid thrust and roll and pitch angles.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (3,) containing the commanded thrust, roll and pitch

        """
        clipped_action = np.clip(action, [0,-np.pi/6, -np.pi/6], [self.MAX_THRUST, np.pi/6, np.pi/6])
        return clipped_action

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Very important

        """
        return {"answer": 42} # You know what this is, right?
