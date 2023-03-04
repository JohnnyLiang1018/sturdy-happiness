import gym
import torch as torch
import numpy as np

class VectorizedGym(gym.Env):

    def __init__(self):
        self.envs = gym.vector.SyncVectorEnv([
            lambda: gym.make("Pendulum-v1"),
            lambda: gym.make("Pendulum-v1"),
            lambda: gym.make("Pendulum-v1")
        ])
        self.num_sim = 2
        self.num_real = 1
        self.observation_space = self.envs.observation_space
        self.action_space = self.envs.action_space
    
    def step(self,actions):
        a_sim = [[float(actions[0])]]
        a_real = [[float(actions[1])]]
        a = np.asarray(a_sim*self.num_sim+a_real*self.num_real)
        obs,reward,termination,info = self.envs.step(a)
        return obs,reward,termination,info

    def reset(self):
        return self.envs.reset()
    
