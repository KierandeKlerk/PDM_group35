import os
import numpy as np
from datetime import datetime
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class RRTAviary(CtrlAviary):
    '''Drone environment derived from gym_pybullet's CtrlAviary with specific purpose of being used with our RRTstar implementation'''

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'):
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder)

    ###############################################################################

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
        
    ######################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, bool]
            Dummy value.

        """
        return {"is_yeet": True} #### GenZ comminucation