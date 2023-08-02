from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import numpy as np

class PhysicalEnv():

    def __init__(self):
        self.observation_space = Box(low=-10, high=160, shape=(5,))
        self.action_space = Box(low=np.array([-180, -180]), high=np.array([180,180]))