from collections import deque, OrderedDict

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.rollout_functions import rollout, multitask_rollout, ensemble_rollout
# from rlkit.samplers.rollout_functions import ensemble_ucb_rollout
from rlkit.samplers.rollout import ensemble_ucb_rollout, ensemble_real_rollout, ensemble_eval_rollout, ensemble_eval
from rlkit.samplers.data_collector.base import PathCollector
import gym

class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            noise_flag=0,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._noise_flag = noise_flag
        
        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                noise_flag=self._noise_flag,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def collect_normalized_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            input_mean,
            input_std,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = normalized_rollout(
                self._env,
                self._policy,
                noise_flag=self._noise_flag,
                max_path_length=max_path_length_this_loop,
                input_mean=input_mean,
                input_std=input_std,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )
    
    
class EnsembleMdpPathCollector(PathCollector):
    def __init__(
            self,
            client,
            env_sim,
            env_real,
            policy,
            num_ensemble,
            noise_flag=0,
            ber_mean=0.5,
            eval_flag=False,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            critic1=None,
            critic2=None,
            inference_type=0.0,
            feedback_type=1,
            use_static_real_replay=True
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self.client = client ##
        self._env = env_sim ##
        self._env_real = env_real ##
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self.num_ensemble = num_ensemble
        self.num_sim = 3 ##
        self.num_real = 3 ##
        self.eval_flag = eval_flag
        self.ber_mean = ber_mean
        self.critic1 = critic1
        self.critic2 = critic2
        self.inference_type = 1
        self.feedback_type = feedback_type
        self._noise_flag = noise_flag
        self.use_static_real_replay = use_static_real_replay
        
        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths_sim = [] ##
        paths_real = [] ##
        paths = []
        num_steps_collected = 0
        print("collecting new path")
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            if self.eval_flag:
                path = ensemble_eval_rollout(
                    self._env,
                    self._policy,
                    self.num_ensemble,
                    max_path_length=max_path_length_this_loop,
                )
            else:
                if self.inference_type > 0: # UCB
                    path_sim = ensemble_ucb_rollout(
                        self._env,
                        self._policy[:self.num_sim], ##
                        critic1=self.critic1[:self.num_sim], ##
                        critic2=self.critic2[:self.num_sim], ##
                        inference_type=self.inference_type,
                        feedback_type=self.feedback_type,
                        num_ensemble=self.num_ensemble,  ##
                        noise_flag=self._noise_flag,
                        max_path_length=max_path_length_this_loop,
                        ber_mean=self.ber_mean,
                    )
                    if self.use_static_real_replay != True:
                        # path_real = ensemble_ucb_rollout(
                        #     self._env_real,
                        #     self._policy[self.num_sim:], ##
                        #     critic1=self.critic1[self.num_sim:], ##
                        #     critic2=self.critic2[self.num_sim:], ##
                        #     inference_type=self.inference_type,
                        #     feedback_type=self.feedback_type,
                        #     num_ensemble=self.num_ensemble, ##
                        #     noise_flag=self._noise_flag,
                        #     max_path_length=max_path_length_this_loop,
                        #     ber_mean=self.ber_mean,
                        # )
                        path_real = ensemble_real_rollout(
                            self._env,
                            self._policy,
                            self.num_ensemble,
                            num_steps,
                            max_path_length_this_loop
                        )
                    else:
                        path_real = None

                # if self.inference_type > 0: # UCB
                #     sim_1_path, sim_2_path, real_path = ensemble_ucb_rollout(
                #         self.client,
                #         self._env,
                #         self._policy,
                #         critic1=self.critic1,
                #         critic2=self.critic2,
                #         inference_type=self.inference_type,
                #         feedback_type=self.feedback_type,
                #         num_ensemble=self.num_ensemble,
                #         noise_flag=self._noise_flag,
                #         max_path_length=max_path_length_this_loop,
                #         ber_mean=self.ber_mean,
                #     )
                    sim = False
                    real = False

                    if(num_steps_collected / 2 > num_steps / 5): ##
                        real = True

                    path_len_1 = len(path_sim['actions'])
                    if(path_len_1 != max_path_length and not path_sim['terminals'][-1] and discard_incomplete_paths):
                        print("sim discard")
                        sim_1 = True
                     
                    if self.use_static_real_replay:
                        real = True

                    else:
                        path_len_2 = len(path_real['actions'])
                        if(path_len_2 != max_path_length and not path_real['terminals'][-1] and discard_incomplete_paths):
                            print("real discard")
                            real = True
            
                    if sim != True:
                        num_steps_collected += path_len_1
                        paths_sim.append(path_sim)
                    if real != True:
                        num_steps_collected += path_len_2
                        paths_real.append(path_real)

                else:
                    path_real = ensemble_real_rollout(
                        self._env,
                        self._policy,
                        self.num_ensemble,
                        num_steps,
                        max_path_length_this_loop
                    )

                    path_sim = None
                    
                    self._num_paths_total += len(path_real)
                    self._num_paths_total += len(paths_real)
                    self._num_steps_total += num_steps
                    self._epoch_paths.extend(paths)

                    return path_sim, path_real

                # else:
                #     path = ensemble_rollout(
                #         self._env,
                #         self._policy,
                #         self.num_ensemble,
                #         noise_flag=self._noise_flag,
                #         max_path_length=max_path_length_this_loop,
                #         ber_mean=self.ber_mean,
                #     )
                #     path_len = len(path['actions'])
                #     if (
                #         path_len != max_path_length
                #         and not path['terminals'][-1]
                #         and discard_incomplete_paths
                #     ):
                #         break
                #     num_steps_collected += path_len
                #     paths.append(path)
                #     self._num_paths_total += len(paths)
                #     self._num_steps_total += num_steps_collected
                #     self._epoch_paths.extend(paths)
                #     return paths

                    
        #     path_len = len(path['actions'])
        #     if (
        #             path_len != max_path_length
        #             and not path['terminals'][-1]
        #             and discard_incomplete_paths
        #     ):
        #         break
        #     num_steps_collected += path_len
        #     paths.append(path)
        # self._num_paths_total += len(paths)
        # self._num_steps_total += num_steps_collected
        # self._epoch_paths.extend(paths)
        # return paths


        self._num_paths_total += len(paths_sim)
        self._num_paths_total += len(paths_real)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths_sim)
        self._epoch_paths.extend(paths_real)    

        return paths_sim, paths_real

    # def collect_new_paths(
    #         self,
    #         max_path_length,
    #         num_steps,
    #         discard_incomplete_paths,
    # ):
    #     paths = []
    #     num_steps_collected = 0
    #     while num_steps_collected < num_steps:
    #         max_path_length_this_loop = min(  # Do not go over num_steps
    #             max_path_length,
    #             num_steps - num_steps_collected,
    #         )
    #         if self.eval_flag:
    #             path = ensemble_eval_rollout(
    #                 self._env,
    #                 self._policy,
    #                 self.num_ensemble,
    #                 max_path_length=max_path_length_this_loop,
    #             )
    #         else:
    #             if self.inference_type > 0: # UCB
    #                 path = ensemble_ucb_rollout(
    #                     self._env,
    #                     self._policy,
    #                     critic1=self.critic1,
    #                     critic2=self.critic2,
    #                     inference_type=self.inference_type,
    #                     feedback_type=self.feedback_type,
    #                     num_ensemble=self.num_ensemble,
    #                     noise_flag=self._noise_flag,
    #                     max_path_length=max_path_length_this_loop,
    #                     ber_mean=self.ber_mean,
    #                 )
    #             else:
    #                 path = ensemble_rollout(
    #                     self._env,
    #                     self._policy,
    #                     self.num_ensemble,
    #                     noise_flag=self._noise_flag,
    #                     max_path_length=max_path_length_this_loop,
    #                     ber_mean=self.ber_mean,
    #                 )
    #         path_len = len(path['actions'])
    #         if (
    #                 path_len != max_path_length
    #                 and not path['terminals'][-1]
    #                 and discard_incomplete_paths
    #         ):
    #             break
    #         num_steps_collected += path_len
    #         paths.append(path)
    #     self._num_paths_total += len(paths)
    #     self._num_steps_total += num_steps_collected
    #     self._epoch_paths.extend(paths)
    #     return paths
    
    def reward_eval(self):
        r_sum = ensemble_eval(self._env, self._policy, self.num_ensemble, max_path_length=1000)
        return r_sum

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )
