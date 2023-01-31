import argparse
from copyreg import pickle
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnsembleEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger_custom, set_seed
from rlkit.samplers.data_collector import EnsembleMdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.neurips20_sac_ensemble import NeurIPS20SACEnsembleTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rollout import ensemble_ucb_rollout
from client import Client
import numpy as np
from gym import spaces
import pickle

class Sunrise(object):

    def __init__(self,obs_dim,action_dim,layer_size,num_layer):
        self.L_qf1 = []
        self.L_qf2 = []
        self.L_target_qf1 = []
        self.L_target_qf2 = []
        self.L_policy = []
        self.L_eval_policy = []
        network_structure = [layer_size] * num_layer

        for _ in range(2):
            qf1 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=network_structure,
            )
            qf2 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=network_structure,
            )
            target_qf1 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=network_structure,
            )
            target_qf2 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=network_structure,
            )
            policy = TanhGaussianPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=network_structure,
            )
            eval_policy = MakeDeterministic(policy)
        
            self.L_qf1.append(qf1)
            self.L_qf2.append(qf2)
            self.L_target_qf1.append(target_qf1)
            self.L_target_qf2.append(target_qf2)
            self.L_policy.append(policy)
            self.L_eval_policy.append(eval_policy)
        
        
    
    def ucb(self):
        d = ensemble_ucb_rollout("env",self.L_policy,self.L_qf1,self.L_qf2,inference_type=0.0,feedback_type=1,num_ensemble=2,ber_mean=0.5)
        return d

    def test(self):
        client = Client()
        policy = pickle.dumps(self.L_qf1[0])
        client.test(policy)


sunrise = Sunrise(3,1,layer_size=256,num_layer=2)

# print(sunrise.ucb())
print(sunrise.test())