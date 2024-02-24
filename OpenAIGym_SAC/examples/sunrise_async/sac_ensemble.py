from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from rlkit.samplers.rollout import ensemble_eval
import gym


class NeurIPS20SACEnsembleTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            env_real,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            num_ensemble,
            feedback_type,
            temperature,
            temperature_act,
            expl_gamma,
            log_dir,
            num_sim, ##
            num_real, ##
        
            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.eval_env = env_real ##
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        
        self.target_update_period = target_update_period
        
        self.num_ensemble = num_sim + num_real
        self.feedback_type = feedback_type
        self.temperature = temperature
        self.temperature_act = temperature_act
        self.expl_gamma = expl_gamma
        self.model_dir = log_dir + '/test1'
        self.num_sim = num_sim ##
        self.num_real = num_real ##
        
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Pendulum action space: (1,)
                self.target_entropy = -np.prod((1,)).item()  # heuristic value from Tuomas
            self.alpha_optimizer, self.log_alpha = [], []
            for _ in range(self.num_ensemble):
                log_alpha = ptu.zeros(1, requires_grad=True)
                alpha_optimizer = optimizer_class(
                    [log_alpha],
                    lr=policy_lr,
                )
                self.alpha_optimizer.append(alpha_optimizer)
                self.log_alpha.append(log_alpha)
                

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduce=False)
        self.vf_criterion = nn.MSELoss(reduce=False)
        
        self.policy_optimizer, self.qf1_optimizer, self.qf2_optimizer, = [], [], []
        
        for en_index in range(self.num_ensemble):
            policy_optimizer = optimizer_class(
                self.policy[en_index].parameters(),
                lr=policy_lr,
            )
            qf1_optimizer = optimizer_class(
                self.qf1[en_index].parameters(),
                lr=qf_lr,
            )
            qf2_optimizer = optimizer_class(
                self.qf2[en_index].parameters(),
                lr=qf_lr,
            )
            self.policy_optimizer.append(policy_optimizer)
            self.qf1_optimizer.append(qf1_optimizer)
            self.qf2_optimizer.append(qf2_optimizer)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self.diagram_statistics = OrderedDict() ##
        self.diagram_statistics.update({'Policy_loss': []}) ##
        self.diagram_statistics.update({'Log_pi': []}) ##
        self.diagram_statistics.update({'R_sum': []}) ##
        self.diagram_statistics.update({'Weight': []}) ##
        self.diagram_statistics.update({'Std_q': []}) ##
        self.diagram_statistics.update({'Log_pi': []}) ##
        self.diagram_statistics.update({"Q_action": []}) ##
        self.diagram_statistics.update({'Critic_loss': []}) ##
        self.diagram_statistics.update({'alpha_logpi': []}) ##
        self.diagram_statistics.update({'alpha': []}) ##
        self.diagram_statistics.update({'R_eval': []}) ##
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def corrective_feedback(self, obs, update_type, is_sim, all_ensemble=False):
        std_Q_list = []
        # obs_sim = obs[:,:3]
        # obs_real = obs[:,3:]
        # print("obs shape ", obs)
        if self.feedback_type == 0  or self.feedback_type == 2:
            for en_index in range(self.num_ensemble):
                with torch.no_grad():
                    policy_action, _, _, _, *_ = self.policy[en_index](
                        obs, reparameterize=True, return_log_prob=True,
                    )
                    if update_type == 0:
                        actor_Q1 = self.qf1[en_index](obs, policy_action)
                        actor_Q2 = self.qf2[en_index](obs, policy_action)
                    else:
                        actor_Q1 = self.target_qf1[en_index](obs, policy_action)
                        actor_Q2 = self.target_qf2[en_index](obs, policy_action)
                    mean_actor_Q= 0.5*(actor_Q1 + actor_Q2)
                    var_Q = 0.5*((actor_Q1 - mean_actor_Q)**2 + (actor_Q2 - mean_actor_Q)**2)
                std_Q_list.append(torch.sqrt(var_Q).detach())
                
        elif self.feedback_type == 1 or self.feedback_type == 3:
            mean_Q, var_Q = None, None
            L_target_Q = []

            num_ensemble = 0
            if all_ensemble:
                ensemble = range(0, self.num_sim+self.num_real)
                num_ensemble = self.num_sim + self.num_real

            elif is_sim:
                ensemble = range(self.num_sim)
                num_ensemble = self.num_sim
            else:
                ensemble = range(self.num_sim, self.num_sim+self.num_real)
                num_ensemble = self.num_real
            # ensemble = range(self.num_ensemble)
            # num_ensemble = self.num_ensemble

            for en_index in ensemble:
                with torch.no_grad():
                    policy_action, _, _, _, *_ = self.policy[en_index](
                        obs, reparameterize=True, return_log_prob=True,
                    )  ## policy action from sim environment?
                    
                    if update_type == 0: # actor
                        target_Q1 = self.qf1[en_index](obs, policy_action)
                        target_Q2 = self.qf2[en_index](obs, policy_action)
                    else: # critic
                        target_Q1 = self.target_qf1[en_index](obs, policy_action)
                        target_Q2 = self.target_qf2[en_index](obs, policy_action)
                    L_target_Q.append(target_Q1)
                    L_target_Q.append(target_Q2)
                    if en_index == ensemble.start:
                        mean_Q = 0.5*(target_Q1 + target_Q2) / num_ensemble
                    else:
                        mean_Q += 0.5*(target_Q1 + target_Q2) / num_ensemble


            for en_index in range(len(L_target_Q)):
                if en_index == 0:
                    ## var_Q = (target_Q.detach() - mean_Q)**2
                    var_Q = (L_target_Q[en_index].detach() - mean_Q)**2
                else:
                    ## var_Q += (target_Q.detach() - mean_Q)**2
                    var_Q += (L_target_Q[en_index].detach() - mean_Q)**2
            var_Q = var_Q / len(L_target_Q)
            std_Q_list.append(torch.sqrt(var_Q).detach())
            # std_Q_list[-1] = torch.tensor(1.0) ##

        return std_Q_list
    
    def corrective_feedback_exp(self, obs, sim_a, update_type, is_sim):
        std_Q_list = []
        # obs_sim = obs[:,:3]
        # obs_real = obs[:,3:]
        # print("obs shape ", obs)
        if self.feedback_type == 0 or self.feedback_type == 2:
            for en_index in range(self.num_ensemble):
                with torch.no_grad():
                    policy_action, _, _, _, *_ = self.policy[en_index](
                        obs, reparameterize=True, return_log_prob=True,
                    )
                    if update_type == 0:
                        actor_Q1 = self.qf1[en_index](obs, policy_action)
                        actor_Q2 = self.qf2[en_index](obs, policy_action)
                    else:
                        actor_Q1 = self.target_qf1[en_index](obs, policy_action)
                        actor_Q2 = self.target_qf2[en_index](obs, policy_action)
                    mean_actor_Q= 0.5*(actor_Q1 + actor_Q2)
                    var_Q = 0.5*((actor_Q1 - mean_actor_Q)**2 + (actor_Q2 - mean_actor_Q)**2)
                std_Q_list.append(torch.sqrt(var_Q).detach())
                
        elif self.feedback_type == 1 or self.feedback_type == 3:
            mean_Q_sim, mean_Q_real, var_Q = None, None, None
            Q_sim = []
            Q_real = []
            L_target_Q = []
            
            ## Sim agent will compute mean Q from the real agent ensemble
            ## and calculate variance for each sim agent
            r = range(self.num_sim)
            r_target = range(self.num_sim, self.num_sim+self.num_real)


            for en_index in r:
                with torch.no_grad():
                    policy_action_sim, _, _, _, *_ = self.policy[en_index](
                        obs, reparameterize=True, return_log_prob=True,
                    )  
                    policy_action_real,_,_,_, *_ = self.policy[en_index+self.num_sim](
                        obs, reparameterize=True, return_log_prob=True,
                    )
                    
                    if update_type == 0: # actor
                        target_Q1_sim = self.qf1[en_index](obs, policy_action_sim)
                        target_Q2_sim = self.qf2[en_index](obs, policy_action_sim)
                        target_Q1_real = self.qf1[en_index+self.num_sim](obs, policy_action_real)
                        target_Q2_real = self.qf2[en_index+self.num_sim](obs, policy_action_real)
                        # target_Q1 = self.qf1[en_index+self.num_sim](obs, policy_action_real)
                        # target_Q2 = self.qf2[en_index+self.num_sim](obs, policy_action_real)
                    else: # critic
                        target_Q1_sim = self.target_qf1[en_index](obs, policy_action_sim)
                        target_Q2_sim = self.target_qf2[en_index](obs, policy_action_sim)
                        target_Q1_real = self.target_qf1[en_index+self.num_sim](obs, policy_action_real)
                        target_Q2_real = self.target_qf2[en_index+self.num_sim](obs, policy_action_real)
                        # target_Q1 = self.target_qf1[en_index+self.num_sim](obs, policy_action_real)
                        # target_Q2 = self.target_qf2[en_index+self.num_sim](obs, policy_action_real)
                    # L_target_Q.append(target_Q1)
                    # L_target_Q.append(target_Q2)
                    Q_sim.append(target_Q1_sim)
                    Q_sim.append(target_Q2_sim)
                    Q_real.append(target_Q1_real)
                    Q_real.append(target_Q2_real)

                    if en_index == r.start:
                        mean_Q_sim = 0.5*(target_Q1_sim + target_Q2_sim) / self.num_sim
                        mean_Q_real = 0.5*(target_Q1_real + target_Q2_real) / self.num_real
                        # mean_Q = 0.5*(target_Q1+target_Q2) / self.num_real
                    else:
                        mean_Q_sim += 0.5*(target_Q1_sim + target_Q2_sim) / self.num_sim
                        mean_Q_real += 0.5*(target_Q1_real + target_Q2_real) / self.num_real
                        # mean_Q += 0.5*(target_Q1+target_Q2) / self.num_real

                    # print("mean Q", np.mean(ptu.get_numpy(mean_Q)))


            for en_index in range(len(L_target_Q)):
                var_Q_sim = (Q_sim[en_index].detach() - mean_Q_sim)**2
                var_Q_real = (Q_real[en_index].detach() - mean_Q_real)**2
                if en_index == 0:
                    # var_Q = (L_target_Q[en_index].detach() - mean_Q)**2
                    ## var_Q = abs(var_Q_sim * var_Q_real)
                    var_Q_sim = (Q_sim[en_index].detach() - mean_Q_sim)**2
                    var_Q_real = (Q_real[en_index].detach() - mean_Q_real)**2
                else:
                    # var_Q += (L_target_Q[en_index].detach() - mean_Q)**2
                    ## var_Q += abs(var_Q_sim * var_Q_real)
                    var_Q_sim += (Q_sim[en_index].detach() - mean_Q_sim)**2
                    var_Q_real += (Q_real[en_index].detach() - mean_Q_real)**2
            
            # var_Q = var_Q / len(L_target_Q)
            var_Q_sim = var_Q_sim / len(Q_sim)
            var_Q_real = var_Q_real / len(Q_real)
            var_Q = var_Q_sim / var_Q_real
            # print("var Q", np.mean(ptu.get_numpy(var_Q)))
            std_Q_list.append(torch.sqrt(var_Q).detach())
                # std_Q_list[-1] = torch.tensor(1.0) ##

        return std_Q_list
        
    def train_from_torch(self, batch):
        torch.autograd.set_detect_anomaly(True)
        # rewards = torch.cat((batch_sim['rewards'],batch_real['rewards']))
        # terminals = torch.cat((batch_sim['terminals'], batch_real['terminals']))
        # obs = torch.cat((batch_sim['observations'], batch_real['observations']))
        # actions = torch.cat((batch_sim['actions'], batch_real['actions']))
        # next_obs = torch.cat((batch_sim['next_observations'], batch_real['next_observations']))
        # masks = torch.cat((batch_sim['masks'], batch_real['masks']))
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        masks = batch['masks']
        
        # variables for logging
        tot_qf1_loss, tot_qf2_loss, tot_q1_pred, tot_q2_pred, tot_q_target = 0, 0, 0, 0, 0
        tot_log_pi, tot_policy_mean, tot_policy_log_std, tot_policy_loss = 0, 0, 0, 0
        tot_alpha, tot_alpha_loss = 0, 0
        
        std_Q_actor_list = self.corrective_feedback(obs=obs, update_type=0)
        std_Q_critic_list = self.corrective_feedback(obs=next_obs, update_type=1)

        # obs_sim = obs[:,:3]
        # obs_real = obs[:,3:]

        # next_sim = next_obs[:,:3]
        # next_real = next_obs[:,3:]
        # log_pi_list = [] ##
        
        for en_index in range(self.num_ensemble):
            mask = masks[:,en_index].reshape(-1, 1)

            """
            Policy and Alpha Loss
            """
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy[en_index](
                obs, reparameterize=True, return_log_prob=True,
            )
            # log_pi_list.append(log_pi) ##

            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[en_index] * (log_pi + self.target_entropy).detach()) * mask
                alpha_loss = alpha_loss.sum() / (mask.sum() + 1)
                self.alpha_optimizer[en_index].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[en_index].step()
                alpha = self.log_alpha[en_index].exp()
            else:
                alpha_loss = 0
                alpha = 1

            q_new_actions = torch.min(
                self.qf1[en_index](obs, new_obs_actions),
                self.qf2[en_index](obs, new_obs_actions),
            )
            # q_new_actions = self.target_qf1[en_index](obs, new_obs_actions)
            # q_new_actions = self.qf1[en_index](obs,new_obs_actions)
            
            if self.feedback_type == 0 or self.feedback_type == 2:
                std_Q = std_Q_actor_list[en_index]
            else:
                std_Q = std_Q_actor_list[0]
                
            if self.feedback_type == 1 or self.feedback_type == 0:
                weight_actor_Q = torch.sigmoid(-std_Q*self.temperature_act) + 0.5
            else:
                weight_actor_Q = 2*torch.sigmoid(-std_Q*self.temperature_act)
            policy_loss = (alpha*log_pi - q_new_actions - self.expl_gamma * std_Q) * mask * weight_actor_Q.detach()
            policy_loss = policy_loss.sum() / (mask.sum() + 1)

            """
            QF Loss
            """
            
            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, _, _, new_log_pi, *_ = self.policy[en_index](
                next_obs, reparameterize=True, return_log_prob=True,
            )
            target_q_values = torch.min(
                self.target_qf1[en_index](next_obs, new_next_actions),
                self.target_qf2[en_index](next_obs, new_next_actions),
            ) - alpha * new_log_pi
            
            if self.feedback_type == 0 or self.feedback_type == 2:
                if self.feedback_type == 0:
                    weight_target_Q = torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature) + 0.5
                else:
                    weight_target_Q = 2*torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature)
            else:
                if self.feedback_type == 1:
                    weight_target_Q = torch.sigmoid(-std_Q_critic_list[0]*self.temperature) + 0.5
                else:
                    weight_target_Q = 2*torch.sigmoid(-std_Q_critic_list[0]*self.temperature)

            ## Compute policy gradient before calling qf again

            self.policy_optimizer[en_index].zero_grad()
            policy_loss.backward()
            self.policy_optimizer[en_index].step()

            q1_pred = self.qf1[en_index](obs, actions)
            q2_pred = self.qf2[en_index](obs, actions)

            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach()) * mask * (weight_target_Q.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach()) * mask * (weight_target_Q.detach())
            qf1_loss = qf1_loss.sum() / (mask.sum() + 1)
            qf2_loss = qf2_loss.sum() / (mask.sum() + 1)
            
            """
            Update networks
            """
            self.qf1_optimizer[en_index].zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer[en_index].step()

            self.qf2_optimizer[en_index].zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer[en_index].step()


            """
            Soft Updates
            """
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf1[en_index], self.target_qf1[en_index], self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.qf2[en_index], self.target_qf2[en_index], self.soft_target_tau
                )
                
            """
            Statistics for log
            """
            tot_qf1_loss += qf1_loss * (1/self.num_ensemble)
            tot_qf2_loss += qf2_loss * (1/self.num_ensemble)
            tot_q1_pred += q1_pred * (1/self.num_ensemble)
            tot_q2_pred += q2_pred * (1/self.num_ensemble)
            tot_q_target += q_target * (1/self.num_ensemble)
            tot_log_pi += log_pi * (1/self.num_ensemble)
            tot_policy_mean += policy_mean * (1/self.num_ensemble)
            tot_policy_log_std += policy_log_std * (1/self.num_ensemble)
            tot_alpha += alpha.item() * (1/self.num_ensemble)
            tot_alpha_loss += alpha_loss.item()
            tot_policy_loss = torch.mean(log_pi - q_new_actions) * (1/self.num_ensemble)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(tot_qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(tot_qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                tot_policy_loss
            ))
            self.diagram_statistics['Policy_loss'].append(np.mean(ptu.get_numpy(
                tot_policy_loss
            ))) ##

            self.diagram_statistics['Weight'].append(np.mean(ptu.get_numpy(weight_target_Q))) ##

            r_sum = ensemble_eval(self.eval_env, self.policy, self.num_ensemble) ##
            self.diagram_statistics['R_sum'].append(r_sum) ##

            # self.diagram_statistics['Log_pi'].append(log_pi_list) ##

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(tot_q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(tot_q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(tot_q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(tot_log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(tot_policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(tot_policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = tot_alpha
                self.eval_statistics['Alpha Loss'] = tot_alpha_loss
                
        self._n_train_steps_total += 1

    def train_from_torch_exp(self, batch_sim, batch_sim_, batch_real, tuning, old_appr, num_epoch):
        # torch.autograd.set_detect_anomaly(True)
        # rewards = torch.cat((batch_sim['rewards'],batch_real['rewards']))
        # terminals = torch.cat((batch_sim['terminals'], batch_real['terminals']))
        # obs = torch.cat((batch_sim['observations'], batch_real['observations']))
        # actions = torch.cat((batch_sim['actions'], batch_real['actions']))
        # next_obs = torch.cat((batch_sim['next_observations'], batch_real['next_observations']))
        # masks = torch.cat((batch_sim['masks'], batch_real['masks']))

        
        
        if tuning == True:
            rewards_sim = batch_sim['rewards']
            rewards_real = batch_real['rewards']
            terminals_sim = batch_sim['terminals']
            terminals_real = batch_real['terminals']
            obs_sim = batch_sim['observations']
            obs_real = batch_real['observations']
            actions_sim = batch_sim['actions']
            actions_real = batch_real['actions']
            next_obs_sim = batch_sim['next_observations']
            next_obs_real = batch_real['next_observations']
            mask_sim = batch_sim['masks']
            mask_real = batch_real['masks']


            # std_Q_actor_list_sim = self.corrective_feedback_exp(obs=obs, update_type=0,is_sim=True)
            # std_Q_actor_list_real = self.corrective_feedback_exp(obs=obs, update_type=0,is_sim=False)
            # std_Q_critic_list_sim = self.corrective_feedback_exp(obs=next_obs, update_type=1, is_sim=True)
            # std_Q_critic_list_real = self.corrective_feedback_exp(obs=next_obs, update_type=1, is_sim=False)

            std_Q_actor_list_sim = self.corrective_feedback(obs=obs_sim, update_type=0,is_sim=True)
            std_Q_critic_list_sim = self.corrective_feedback(obs=next_obs_sim, update_type=1, is_sim=True)
            std_Q_actor_list_real = self.corrective_feedback(obs=obs_real, update_type=0,is_sim=False)
            std_Q_critic_list_real = self.corrective_feedback(obs=next_obs_real, update_type=1, is_sim=False)

        else:
            rewards_sim = torch.cat((batch_sim_['rewards'],batch_sim['rewards']))
            rewards_real = torch.cat((batch_sim['rewards'],batch_real['rewards']))
            terminals_sim = torch.cat((batch_sim_['terminals'], batch_sim['terminals']))
            terminals_real = torch.cat((batch_sim['terminals'], batch_real['terminals']))
            obs_sim  = torch.cat((batch_sim_['observations'], batch_sim['observations']))
            obs_real = torch.cat((batch_sim['observations'], batch_real['observations']))
            actions_sim = torch.cat((batch_sim_['actions'], batch_sim['actions']))
            actions_real = torch.cat((batch_sim['actions'], batch_real['actions']))
            next_obs_sim = torch.cat((batch_sim_['next_observations'], batch_sim['next_observations']))
            next_obs_real = torch.cat((batch_sim['next_observations'], batch_real['next_observations']))
            mask_sim = torch.cat((batch_sim_['masks'], batch_sim['masks']))
            mask_real = torch.cat((batch_sim['masks'], batch_real['masks']))

            # rewards_sim = batch_sim['rewards']
            # rewards_real = batch_real['rewards']
            # terminals_sim = batch_sim['terminals']
            # terminals_real = batch_real['terminals']
            # obs_sim = batch_sim['observations']
            # obs_real = batch_real['observations']
            # actions_sim = batch_sim['actions']
            # actions_real = batch_real['actions']



            ## TODO 
            std_Q_actor_list_sim = self.corrective_feedback(obs=obs_sim, update_type=0, is_sim=True, all_ensemble=True)
            std_Q_critic_list_sim = self.corrective_feedback(obs=next_obs_sim, update_type=1, is_sim=True, all_ensemble=True)


            if old_appr == True:
                std_Q_actor_list_real = self.corrective_feedback(obs=obs_real, update_type=0, is_sim=False)
                std_Q_critic_list_real = self.corrective_feedback(obs=next_obs_real, update_type=1, is_sim=False)
            else:
                std_Q_actor_list_sim_ = self.corrective_feedback(obs=batch_sim['observations'], update_type=0, is_sim=True)
                std_Q_critic_list_sim_ = self.corrective_feedback_exp(obs=batch_sim['next_observations'], sim_a=batch_sim['actions'], update_type=1, is_sim=True)
                std_Q_actor_list_real_ = self.corrective_feedback(obs=batch_real['observations'], update_type=0, is_sim=False)
                std_Q_critic_list_real_ = self.corrective_feedback(obs=batch_real['next_observations'], update_type=1, is_sim=False)
                
                std_Q_actor_list_real = [torch.cat((std_Q_actor_list_sim_[0], std_Q_actor_list_real_[0]))]
                std_Q_critic_list_real = [torch.cat((std_Q_critic_list_sim_[0], std_Q_critic_list_real_[0]))]
 
            # std_Q_actor_list = self.corrective_feedback(obs=obs, update_type=0,is_sim=True)
            # std_Q_critic_list = self.corrective_feedback(obs=next_obs, update_type=1, is_sim=True)


        # variables for logging
        tot_qf1_loss, tot_qf2_loss, tot_q1_pred, tot_q2_pred, tot_q_target = 0, 0, 0, 0, 0
        tot_log_pi, tot_policy_mean, tot_policy_log_std, tot_policy_loss, tot_real_policy_loss, tot_real_qf_loss = 0, 0, 0, 0, 0, 0
        tot_alpha, tot_alpha_loss = 0, 0

        # obs_sim = obs[:,:3]
        # obs_real = obs[:,3:]

        # next_sim = next_obs[:,:3]
        # next_real = next_obs[:,3:]
        # log_pi_list = [] ##
        
        for en_index in range(self.num_ensemble):
            

            """
            Policy and Alpha Loss
            """
            # if tuning == True:
            #     if en_index < self.num_sim:
            #         std_Q_actor_list = std_Q_actor_list_sim
            #         std_Q_critic_list = std_Q_critic_list_sim

            #     else:
            #         std_Q_actor_list = std_Q_actor_list_real
            #         std_Q_critic_list = std_Q_critic_list_real

            # else:
                # if en_index < self.num_sim:
                #     obs = obs_sim
                #     actions = actions_sim
                #     rewards = rewards_sim
                #     next_obs = next_obs_sim
                #     terminals = terminals_sim
                #     masks = mask_sim
                #     std_Q_actor_list = std_Q_actor_list_sim
                #     std_Q_critic_list = std_Q_critic_list_sim

                # else:
                #     obs = obs_real
                #     actions = actions_real
                #     rewards = rewards_real
                #     next_obs = next_obs_real
                #     terminals = terminals_real
                #     masks = mask_real
                #     std_Q_actor_list = std_Q_actor_list_real
                #     std_Q_critic_list = std_Q_critic_list_real

            if en_index < self.num_sim:
                obs = obs_sim
                actions = actions_sim
                rewards = rewards_sim
                next_obs = next_obs_sim
                terminals = terminals_sim
                masks = mask_sim
                std_Q_actor_list = std_Q_actor_list_sim
                std_Q_critic_list = std_Q_critic_list_sim

            else:
                obs = obs_real
                actions = actions_real
                rewards = rewards_real
                next_obs = next_obs_real
                terminals = terminals_real
                masks = mask_real
                std_Q_actor_list = std_Q_actor_list_real
                std_Q_critic_list = std_Q_critic_list_real

            
            mask = masks[:,(en_index%2)].reshape(-1, 1)

            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy[en_index](
                obs, reparameterize=True, return_log_prob=True,
            )

            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[en_index] * (log_pi + self.target_entropy).detach()) * mask
                alpha_loss = alpha_loss.sum() / (mask.sum() + 1)
                self.alpha_optimizer[en_index].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[en_index].step()
                alpha = self.log_alpha[en_index].exp()
            else:
                alpha_loss = 0
                alpha = 1

            q_new_actions = torch.min(
                self.qf1[en_index](obs, new_obs_actions),
                self.qf2[en_index](obs, new_obs_actions),
            )
            
            if self.feedback_type == 0 or self.feedback_type == 2:
                std_Q = std_Q_actor_list[en_index]
            else:
                std_Q = std_Q_actor_list[0]
            
            if self.feedback_type == 1 or self.feedback_type == 0:
                weight_actor_Q = (torch.sigmoid(-std_Q*self.temperature_act) + 0.5)
            else:
                weight_actor_Q = (2*torch.sigmoid(-std_Q*self.temperature_act))
            # print("Weight_actor_Q", weight_actor_Q)
            policy_loss = (alpha*log_pi - q_new_actions - self.expl_gamma * std_Q) * mask * weight_actor_Q.detach()
            policy_loss = policy_loss.sum() / (mask.sum() + 1)

            """
            QF Loss
            """

            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, _, _, new_log_pi, *_ = self.policy[en_index](
                next_obs, reparameterize=True, return_log_prob=True,
            )
            target_q_values = torch.min(
                self.target_qf1[en_index](next_obs, new_next_actions),
                self.target_qf2[en_index](next_obs, new_next_actions),
            ) - alpha * new_log_pi

            # if tuning != True:
            #     std_Q_critic_list = [x * 0 for x in std_Q_critic_list]
            
            if self.feedback_type == 0 or self.feedback_type == 2:
                if self.feedback_type == 0:
                    weight_target_Q = torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature) + 0.5
                else:
                    weight_target_Q = 2*torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature)
            else:
                if self.feedback_type == 1:
                    # print("std Q", np.mean(ptu.get_numpy(std_Q_critic_list[0])))
                    if old_appr:
                        weight_target_Q = torch.sigmoid(-std_Q_critic_list[0]*self.temperature) + 0.5  ##
                    else:
                        weight_target_Q = torch.sigmoid(-std_Q_critic_list[0]*self.temperature) + 0.5  ##
                    # print("weight Q", np.mean(ptu.get_numpy(weight_target_Q)))
                else:
                    weight_target_Q = 2*torch.sigmoid(-std_Q_critic_list[0]*self.temperature)

            self.policy_optimizer[en_index].zero_grad()
            policy_loss.backward()
            self.policy_optimizer[en_index].step()

            q1_pred = self.qf1[en_index](obs, actions)
            q2_pred = self.qf2[en_index](obs, actions)
            
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach()) * mask * (weight_target_Q.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach()) * mask * (weight_target_Q.detach())
            qf1_loss = qf1_loss.sum() / (mask.sum() + 1)
            qf2_loss = qf2_loss.sum() / (mask.sum() + 1)
            
            """
            Update networks
            """
            self.qf1_optimizer[en_index].zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer[en_index].step()

            self.qf2_optimizer[en_index].zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer[en_index].step()

            """
            Soft Updates
            """
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf1[en_index], self.target_qf1[en_index], self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.qf2[en_index], self.target_qf2[en_index], self.soft_target_tau
                )
                
            """
            Statistics for log
            """
            tot_qf1_loss += qf1_loss * (1/self.num_ensemble)
            tot_qf2_loss += qf2_loss * (1/self.num_ensemble)
            tot_q1_pred += q1_pred * (1/self.num_ensemble)
            tot_q2_pred += q2_pred * (1/self.num_ensemble)
            tot_q_target += q_target * (1/self.num_ensemble)
            tot_log_pi += log_pi * (1/self.num_ensemble)
            tot_policy_mean += policy_mean * (1/self.num_ensemble)
            tot_policy_log_std += policy_log_std * (1/self.num_ensemble)
            tot_alpha += alpha.item() * (1/self.num_ensemble)
            tot_alpha_loss += alpha_loss.item()
            tot_policy_loss = (log_pi - q_new_actions).mean() * (1/self.num_ensemble)

            ## stat for real ensemble
            if en_index >= self.num_sim:
                tot_real_policy_loss += (log_pi - q_new_actions).mean() * (1/self.num_real)
                tot_real_qf_loss += (qf1_loss + qf2_loss).mean() / 2 * (1/self.num_real)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(tot_qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(tot_qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                tot_real_policy_loss
            ))
            self.diagram_statistics['Policy_loss'].append(np.mean(ptu.get_numpy(
                tot_real_policy_loss
            ))) ##
            self.diagram_statistics['Critic_loss'].append(np.mean(ptu.get_numpy(
                tot_real_qf_loss
            )))
            self.diagram_statistics['Weight'].append(np.mean(ptu.get_numpy(weight_target_Q))) ##

            r_avg_sim, r_eval = ensemble_eval(self.eval_env, self.policy, self.num_ensemble, num_epoch, max_path_length=100) ##
            self.diagram_statistics['R_sum'].append(r_avg_sim) ##
            if r_eval is not None:
                self.diagram_statistics['R_eval'].append(r_eval) ##
            self.diagram_statistics['Std_q'].append(np.mean(ptu.get_numpy(self.expl_gamma * std_Q))) ##
            self.diagram_statistics['Q_action'].append(np.mean(ptu.get_numpy(q_new_actions))) ##
            self.diagram_statistics['Log_pi'].append(np.mean(ptu.get_numpy(alpha*log_pi))) ##

            # self.diagram_statistics['Log_pi'].append(log_pi_list) ##

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(tot_q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(tot_q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(tot_q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(tot_log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(tot_policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(tot_policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = tot_alpha
                self.eval_statistics['Alpha Loss'] = tot_alpha_loss

            if r_eval is not None and r_eval > 10:
                self.save_models(len(self.diagram_statistics['R_sum']))
                
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def get_diagram_diagnostics(self):
        return self.diagram_statistics ##

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
    
    @property
    def networks(self):
        output = []
        for en_index in range(self.num_ensemble):
            output.append(self.policy[en_index])
            output.append(self.qf1[en_index])
            output.append(self.qf2[en_index])
            output.append(self.target_qf1[en_index])
            output.append(self.target_qf2[en_index])
        return output

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
    
    def save_models(self, step):
        for en_index in range(self.num_ensemble):
            # torch.save(
            #     self.policy[en_index].state_dict(), '%s/%d_th_actor_%s.pt' % (self.model_dir, en_index, step)
            # )
            # torch.save(
            #     self.qf1[en_index].state_dict(), '%s/%d_th_1st_critic_%s.pt' % (self.model_dir, en_index, step)
            # )
            # torch.save(
            #     self.qf2[en_index].state_dict(), '%s/%d_th_2nd_critic_%s.pt' % (self.model_dir, en_index, step)
            # )
            # torch.save(
            #     self.target_qf1[en_index].state_dict(), '%s/%d_th_1st_target_critic_%s.pt' % (self.model_dir, en_index, step)
            # )
            # torch.save(
            #     self.target_qf2[en_index].state_dict(), '%s/%d_th_2nd_target_critic_%s.pt' % (self.model_dir, en_index, step)
            # )
            torch.save({
                'policy': self.policy[en_index].state_dict(),
                'qf1': self.qf1[en_index].state_dict(),
                'qf2': self.qf2[en_index].state_dict(),
                'target_qf1': self.target_qf1[en_index].state_dict(),
                'target_qf2': self.target_qf2[en_index].state_dict()
            }, '%s/%d_th_network_dict_%s.pt' % (self.model_dir, en_index, step))
            
    def load_models(self, step):
        for en_index in range(self.num_ensemble):
            # self.policy[en_index].load_state_dict(
            #     torch.load('%s/%d_th_actor_%s.pt' % (self.model_dir, en_index, step))
            # )
            # self.qf1[en_index].load_state_dict(
            #     torch.load('%s/%d_th_1st_critic_%s.pt' % (self.model_dir, en_index, step))
            # )
            # self.qf2[en_index].load_state_dict(
            #     torch.load('%s/%d_th_2nd_critic_%s.pt' % (self.model_dir, en_index, step))
            # )
            # self.target_qf1[en_index].load_state_dict(
            #     torch.load('%s/%d_th_1st_target_critic_%s.pt' % (self.model_dir, en_index, step))
            # )
            # self.target_qf2[en_index].load_state_dict(
            #     torch.load('%s/%d_th_2nd_target_critic_%s.pt' % (self.model_dir, en_index, step))
            # )
            checkpoint = torch.load('%s/%d_th_network_dict_%s.pt' % (self.model_dir, en_index, step))
            self.policy[en_index].load_state_dict(checkpoint['policy'])
            self.qf1[en_index].load_state_dict(checkpoint['qf1'])
            self.qf2[en_index].load_state_dict(checkpoint['qf2'])
            self.target_qf1[en_index].load_state_dict(checkpoint['target_qf1'])
            self.target_qf2[en_index].load_state_dict(checkpoint['target_qf2'])
            
    def print_model(self):
        for name, param in self.policy[0].named_parameters():
            if param.requires_grad:
                print(name)
                print(param.data)
                break