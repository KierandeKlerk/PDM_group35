## PID control class inspired by BaseControl.py and SimplePIDControl.py from gym_pybullet_drones

import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel

class pointmassPID (object):

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        #### Set general use constants #############################
        self.DRONE_MODEL = drone_model # Store the drone model
        self.GRAVITY = g*self._getURDFParameter('m') # Store the weight of the drone
        self.THRUST2WEIGHT = self._getURDFParameter('thrust2weight') 
        self.MAX_THRUST = self.THRUST2WEIGHT*self.GRAVITY # Calculate the maximum thrust the drone can exert
        self.MAX_ROLL_PITCH = np.pi/6 # Set a limit to the pitch and roll of the quadrotor

        # Set PID values for Thrust, Roll and Pitch respectively
        self.P_COEFF_T = np.array([1.0, 1.0, 1.0])
        self.I_COEFF_T = np.array([1.0, 1.0, 1.0])
        self.D_COEFF_T = np.array([1.0, 1.0, 1.0])

        self.reset()


    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        self.control_counter = 0

        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos
                                ):
        """Interface method using `computeControl`.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (16,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        """
        cur_pos=state[0:3]
        cur_quat=state[3:7]
        
        self.control_counter += 1

        action = self._computePID(control_timestep, 
                                  cur_pos, 
                                  target_pos)

        return action

    ################################################################################

    def _computePID(self,
                    control_timestep,
                    cur_pos,
                    target_pos):

        pos_e = target_pos - np.array(cur_pos).reshape(3) # Calculate proportional error
        d_pos_e = (pos_e - self.last_pos_e) / control_timestep # Calculate derivative error
        self.last_pos_e = pos_e # Store proportional error for next control step
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep # Calculate integral error
        
        
        ### Calculate target force vector
        target_force = np.array([0, 0, self.GRAVITY]) \
                       + np.multiply(self.P_COEFF_T, pos_e) \
                       + np.multiply(self.I_COEFF_T, self.integral_pos_e) \
                       + np.multiply(self.D_COEFF_T, d_pos_e)
        

        ### Calculate target orientation
        target_rpy = np.zeros(3)
        sign_z =  np.sign(target_force[2])
        if sign_z == 0:
            sign_z = 1
        target_rpy[0] = np.arcsin(-sign_z*target_force[1] / np.linalg.norm(target_force))
        target_rpy[1] = np.arctan2(sign_z*target_force[0], sign_z*target_force[2])
        target_rpy[2] = 0.

        ### Clip thrust and orientation
        action = np.zeros(3)
        action = np.clip(np.append(np.linalg.norm(target_force), target_rpy[:2]), 
                         [0,-self.MAX_ROLL_PITCH, -self.MAX_ROLL_PITCH], 
                         [self.MAX_THRUST, self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH]) # Clip the action according to the max thrust and the min and max roll and pitch

        return action

    ################################################################################
    
    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        #### Get the XML tree of the drone model to control ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+URDF)
        URDF_TREE = etxml.parse(path).getroot()
        #### Find and return the desired parameter #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]
