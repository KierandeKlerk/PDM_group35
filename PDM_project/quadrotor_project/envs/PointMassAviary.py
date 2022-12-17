import numpy as np
import pybullet as p
import pybullet_data
import gym


class PointMassAviary(gym.Env):
    def init(self):
        super(PointMassAviary, self).__init__()