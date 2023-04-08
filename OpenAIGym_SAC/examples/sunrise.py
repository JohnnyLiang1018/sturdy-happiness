import argparse
import rlkit.torch.pytorch_util as ptu

from rlkit.data_management.env_replay_buffer import EnsembleEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger_custom, set_seed
from rlkit.samplers.data_collector import EnsembleMdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from examples.sunrise_async.VectorizedGym import VectorizedGym
# from rlkit.torch.sac.neurips20_sac_ensemble import NeurIPS20SACEnsembleTrainer
from examples.sunrise_async.sac_ensemble import NeurIPS20SACEnsembleTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import gym
from examples.sunrise_async.client import Client
import pickle
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    # architecture
    parser.add_argument('--num_layer', default=2, type=int)
    
    # train
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--save_freq', default=0, type=int)

    # misc
    parser.add_argument('--seed', default=1, type=int)
    
    # env
    parser.add_argument('--env', default="halfcheetah_poplin", type=str)
    
    # ensemble
    parser.add_argument('--num_ensemble', default=3, type=int)
    parser.add_argument('--ber_mean', default=0.5, type=float)
    
    # inference
    parser.add_argument('--inference_type', default=1.0, type=float)
    
    # corrective feedback
    parser.add_argument('--temperature', default=20.0, type=float)
    
    args = parser.parse_args()
    return args

def get_env(env_name, seed):

    if env_name in ['gym_walker2d', 'gym_hopper',
                    'gym_cheetah', 'gym_ant']:
        from mbbl_env.env.gym_env.walker import env
    env = env(env_name=env_name, rand_seed=seed, misc_info={'reset_type': 'gym'})
    return env

def experiment(variant):
    # expl_env = NormalizedBoxEnv(get_env(variant['env'], variant['seed']))
    # eval_env = NormalizedBoxEnv(get_env(variant['env'], variant['seed']))
    # obs_dim = expl_env.observation_space.low.size
    # action_dim = eval_env.action_space.low.size

    ## num_sim must equal to num_real 
    num_sim = 3
    num_real = 3

    expl_env = VectorizedGym()
    expl_env_sim = gym.make("Pendulum-v1", g=9.7)
    expl_env_real = gym.make("Pendulum-v1", g=9.8)
    obs_dim = 3
    action_dim = 1
    # obs_dim = 5
    # action_dim = 2
    
    M = variant['layer_size']
    num_layer = variant['num_layer']
    network_structure = [M] * num_layer
    
    NUM_ENSEMBLE = variant['num_ensemble']
    L_qf1, L_qf2, L_target_qf1, L_target_qf2, L_policy, L_eval_policy = [], [], [], [], [], []

    # client = Client()
    client = None
    
    for _ in range(NUM_ENSEMBLE*2):
    
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

        qf1.to(ptu.device)
        qf2.to(ptu.device)
        target_qf1.to(ptu.device)
        target_qf2.to(ptu.device)
        policy.to(ptu.device)
        
        L_qf1.append(qf1)
        L_qf2.append(qf2)
        L_target_qf1.append(target_qf1)
        L_target_qf2.append(target_qf2)
        L_policy.append(policy)
        L_eval_policy.append(eval_policy)
    
    eval_path_collector = EnsembleMdpPathCollector(
        client,
        expl_env_sim, ##
        expl_env_real, ##
        L_policy,
        NUM_ENSEMBLE,
        ber_mean=variant['ber_mean'],
        eval_flag=False,
        critic1=L_qf1,
        critic2=L_qf2,
        inference_type=variant['inference_type'],
        feedback_type=1,
    )
    
    expl_path_collector = EnsembleMdpPathCollector(
        client,
        expl_env_sim, ##
        expl_env_real,  ##
        L_policy,
        NUM_ENSEMBLE,
        ber_mean=variant['ber_mean'],
        eval_flag=False,
        critic1=L_qf1,
        critic2=L_qf2,
        inference_type=variant['inference_type'],
        feedback_type=1,
    )
    
    replay_buffer_sim = EnsembleEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env_sim, 
        NUM_ENSEMBLE,
        log_dir=variant['log_dir'],
    )

    replay_buffer_real = EnsembleEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env_real,
        NUM_ENSEMBLE,
        log_dir=variant['log_dir'],
    )
    
    trainer = NeurIPS20SACEnsembleTrainer(
        env=expl_env_sim,
        env_real = expl_env_real,
        policy=L_policy,
        qf1=L_qf1,
        qf2=L_qf2,
        target_qf1=L_target_qf1,
        target_qf2=L_target_qf2,
        num_ensemble=NUM_ENSEMBLE,
        feedback_type=1,
        temperature=variant['temperature'],
        temperature_act=0,
        expl_gamma=0,
        log_dir=variant['log_dir'],
        num_sim = num_sim, ##
        num_real = num_real, ## 
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env_sim,
        evaluation_env=expl_env_real,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer_sim,
        **variant['algorithm_kwargs'],
        replay_buffer_real=replay_buffer_real ##
    )
    
    algorithm.to(ptu.device)
    algorithm.train()
    with open('stat_stable.pickle','wb') as handle:
        pickle.dump(trainer.get_diagram_diagnostics(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dumps(L_policy[0])
    # print("success")


if __name__ == "__main__":
    args = parse_args()
    
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=200,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            save_frequency=args.save_freq,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-3,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        num_ensemble=3,
        num_layer=2,
        seed=args.seed,
        ber_mean=args.ber_mean,
        env=args.env,
        inference_type=1,
        temperature=20,
        log_dir="",
    )
    
                            
    set_seed(args.seed)
    exp_name = 'SUNRISE'
    log_dir = setup_logger_custom(exp_name, variant=variant)
            
    variant['log_dir'] = log_dir
    ptu.set_gpu_mode(True, True)
    print(sys.version)
    experiment(variant)

    