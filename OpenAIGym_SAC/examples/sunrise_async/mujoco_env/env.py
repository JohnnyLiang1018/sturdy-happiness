import gym
from gym import spaces
import numpy as np
import mujoco_py

class MyMujocoEnv(gym.Env):
    def __init__(self):
        self.model = mujoco_py.load_model_from_path('path/to/model.xml')
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)

        high = np.array([np.inf] * self.observation_space.shape[0])
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

    def step(self, action):
        self.sim.data.ctrl[0] = action
        self.sim.step()
        obs = self.sim.data.qpos
        reward = 0
        done = False
        return obs, reward, done, {}

    def reset(self):
        self.sim.reset()
        obs = self.sim.data.qpos
        return obs

    def render(self, mode='human'):
        self.viewer.render()

