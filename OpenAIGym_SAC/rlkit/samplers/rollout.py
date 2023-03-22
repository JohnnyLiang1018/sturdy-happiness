from statistics import mean
import numpy as np
import torch 
from rlkit.torch import pytorch_util as ptu
from examples.sunrise_async.collection_request import CollectionRequest

def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
        env,
        agent,
        noise_flag=0,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        if noise_flag == 1:
            r += np.random.normal(0,1,1)[0]
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def ensemble_rollout(
        env,
        agent,
        num_ensemble,
        noise_flag=0,
        max_path_length=10000,
        ber_mean=0.5,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    masks = [] # mask for bootstrapping
    o = env.reset()
    en_index = np.random.randint(num_ensemble)
    agent[en_index].reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent[en_index].get_action(o)
        next_o, r, d, env_info = env.step(a)
        if noise_flag == 1:
            r += np.random.normal(0,1,1)[0]
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        mask = torch.bernoulli(torch.Tensor([ber_mean]*num_ensemble))
        if mask.sum() == 0:
            rand_index = np.random.randint(num_ensemble, size=1)
            mask[rand_index] = 1
        mask = mask.numpy()
        masks.append(mask)
  
        path_length += 1
        if d.any(): ##
            break
        o = next_o
        if render:
            env.render(**render_kwargs)
    
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    masks = np.array(masks)

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        masks=masks,
    )


def get_ucb_std(obs, policy_action, inference_type, critic1, critic2, 
                feedback_type, en_index, num_ensemble, num_subset, sim_flag):
    obs = ptu.from_numpy(obs).float()
    policy_action = ptu.from_numpy(policy_action).float()
    obs = obs.reshape(1,-1)
    policy_action = policy_action.reshape(1,-1)

    if sim_flag == True:
        ensemble = range(num_subset)
    else:
        ensemble = range(num_ensemble-num_subset, num_ensemble)
    
    if feedback_type == 0 or feedback_type==2:
        with torch.no_grad():
            target_Q1 = critic1[en_index](obs, policy_action)
            target_Q2 = critic2[en_index](obs, policy_action)
        mean_Q = 0.5*(target_Q1.detach() + target_Q2.detach())
        var_Q = 0.5*((target_Q1.detach() - mean_Q)**2 + (target_Q2.detach() - mean_Q)**2)
        ucb_score = mean_Q + inference_type * torch.sqrt(var_Q).detach()

    elif feedback_type == 1 or feedback_type==3:
        mean_Q, var_Q = None, None
        L_target_Q = []
        for en_index in ensemble:
            with torch.no_grad():
                target_Q1 = critic1[en_index](obs, policy_action)
                target_Q2 = critic2[en_index](obs, policy_action)
                L_target_Q.append(target_Q1)
                L_target_Q.append(target_Q2)
                if en_index == ensemble.start:
                    mean_Q = 0.5*(target_Q1 + target_Q2) / num_subset
                else:
                    mean_Q += 0.5*(target_Q1 + target_Q2) / num_subset

        temp_count = 0
        for target_Q in L_target_Q:
            if temp_count == 0:
                var_Q = (target_Q.detach() - mean_Q)**2
            else:
                var_Q += (target_Q.detach() - mean_Q)**2
            temp_count += 1
        var_Q = var_Q / temp_count
        ucb_score = mean_Q + inference_type * torch.sqrt(var_Q).detach()
        
    return ucb_score

# def get_ucb_std(obs, policy_action, inference_type, critic1, critic2, 
#                 feedback_type, en_index, num_ensemble):
#     obs = ptu.from_numpy(obs).float()
#     policy_action = ptu.from_numpy(policy_action).float()
#     obs = obs.reshape(1,-1)
#     policy_action = policy_action.reshape(1,-1)
    
#     if feedback_type == 0 or feedback_type==2:
#         with torch.no_grad():
#             target_Q1 = critic1[en_index](obs, policy_action)
#             target_Q2 = critic2[en_index](obs, policy_action)
#         mean_Q = 0.5*(target_Q1.detach() + target_Q2.detach())
#         var_Q = 0.5*((target_Q1.detach() - mean_Q)**2 + (target_Q2.detach() - mean_Q)**2)
#         ucb_score = mean_Q + inference_type * torch.sqrt(var_Q).detach()

#     elif feedback_type == 1 or feedback_type==3:
#         mean_Q, var_Q = None, None
#         L_target_Q = []
#         for en_index in range(num_ensemble):
#             with torch.no_grad():
#                 target_Q1 = critic1[en_index](obs, policy_action)
#                 target_Q2 = critic2[en_index](obs, policy_action)
#                 L_target_Q.append(target_Q1)
#                 L_target_Q.append(target_Q2)
#                 if en_index == 0:
#                     mean_Q = 0.5*(target_Q1 + target_Q2) / num_ensemble
#                 else:
#                     mean_Q += 0.5*(target_Q1 + target_Q2) / num_ensemble

#         temp_count = 0
#         for target_Q in L_target_Q:
#             if temp_count == 0:
#                 var_Q = (target_Q.detach() - mean_Q)**2
#             else:
#                 var_Q += (target_Q.detach() - mean_Q)**2
#             temp_count += 1
#         var_Q = var_Q / temp_count
#         ucb_score = mean_Q + inference_type * torch.sqrt(var_Q).detach()
        
#     return ucb_score
    
def ensemble_ucb_rollout(
        client,
        env,
        agent,
        critic1,
        critic2,
        inference_type,
        feedback_type,
        num_ensemble,
        noise_flag=0,
        max_path_length=10000,
        ber_mean=0.5,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """

    if render_kwargs is None:
        render_kwargs = {}
    observations_sim_1 = []
    observations_sim_2 = []
    observations_real = []
    actions_sim_1 = []
    actions_sim_2 = []
    actions_real = []
    rewards_sim_1 = []
    rewards_sim_2 = []
    rewards_real = []
    terminals_sim_1 = []
    terminals_sim_2 = []
    terminals_real = []
    agent_infos = []
    env_infos = []

    masks = [] # mask for bootstrapping
    # o = env.reset()
    # o_sim = [client.state_init[0], client.state_init[1]] ##
    # o_real = [client.state_init[2]] ##
    env_init = env.reset() ##
    o_sim = [env_init[0], env_init[1]] ##
    o_real = [env_init[2]] ##

    for en_index in range(num_ensemble):
        agent[en_index].reset()
    next_o = None
    path_length = 0
    # if render:
    #     env.render(**render_kwargs)
    num_sim = 2
    num_real = 1
        
    while path_length < max_path_length:
        a_max_sim, a_max_real, ucb_max_sim, ucb_max_real, agent_info_max_sim, agent_info_max_real = None, None, None, None, None, None
        l_a_sim, l_a_real = [], []
        for sub_env in o_sim:
            for en_index in range(num_sim):
                # _a, agent_info = agent[en_index].get_action(o) 
                _a_sim,  agent_info_sim = agent[en_index].get_action(sub_env) 
                ucb_score_sim = get_ucb_std(sub_env, _a_sim, inference_type, critic1, critic2,
                                    feedback_type, en_index, num_ensemble, num_sim,True)
            
                if en_index == 0:
                    a_max_sim = _a_sim
                    ucb_max_sim = ucb_score_sim
                    agent_info_max_sim = agent_info_sim
                else:
                    if ucb_score_sim > ucb_max_sim:
                        ucb_max_sim = ucb_score_sim
                        a_max_sim = _a_sim
                        agent_info_max_sim = agent_info_sim
            l_a_sim.append(a_max_sim)

        for sub_env in o_real:
            for en_index in range(num_real):
                _a_real, agent_info_real = agent[num_sim+en_index].get_action(sub_env)
                ucb_score_real = get_ucb_std(sub_env, _a_real, inference_type, critic1, critic2,
                                    feedback_type, en_index, num_ensemble, num_real,False)
                if en_index == 0:
                    a_max_real = _a_real
                    ucb_max_real = ucb_score_real
                    agent_info_max_real = agent_info_real
                else:
                    if ucb_score_real > ucb_max_real:
                        ucb_max_real = ucb_score_real
                        a_max_real = _a_real
                        agent_info_max_real = agent_info_real
            l_a_real.append(a_max_real)

        next_o, r, d, env_info = env.step(l_a_sim+l_a_real)
        # next_o, r, d ,env_info = client.request(l_a_sim+l_a_real)
        next_o_sim_1, next_o_sim_2, next_o_real = next_o[0], next_o[1], next_o[2]
        if noise_flag == 1:
            r += np.random.normal(0,1,1)[0]
        # observations.append(np.concatenate((o_sim,o_real)))
        observations_real.append(o_real[0])
        observations_sim_1.append(o_sim[0])
        observations_sim_2.append(o_sim[1])
        rewards_real.append(r[2])
        rewards_sim_1.append(r[0])
        rewards_sim_2.append(r[1])
        terminals_real.append(d[2])
        terminals_sim_1.append(d[0])
        terminals_sim_2.append(d[1])
        actions_real.append(l_a_real[0])
        actions_sim_1.append(l_a_sim[0])
        actions_sim_2.append(l_a_sim[1])
        agent_info = {}
        for key in agent_info_max_sim:
            agent_info[key+"_sim"] = agent_info_max_sim[key]
        for key in agent_info_max_real:
            agent_info[key+"_real"] = agent_info_max_real[key]
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        mask = torch.bernoulli(torch.Tensor([ber_mean]*num_ensemble))
        if mask.sum() == 0:
            rand_index = np.random.randint(num_ensemble, size=1)
            mask[rand_index] = 1
        mask = mask.numpy()
        masks.append(mask)
  
        path_length += 1
        if any(d):
            break
        o_sim = [next_o_sim_1, next_o_sim_2] ##
        o_real = [next_o_real] ##
        # if render:
        #     env.render(**render_kwargs)
    
    actions_real = np.array(actions_real)
    actions_sim_1 = np.array(actions_sim_1)
    actions_sim_2 = np.array(actions_sim_2)
    if len(actions_real.shape) == 1:
        actions_sim_1 = np.expand_dims(actions_sim_1 ,1)
        actions_sim_2 = np.expand_dims(actions_sim_2 ,1)
        actions_real = np.expand_dims(actions_real, 1)

    observations_real = np.array(observations_real)
    observations_sim_1 = np.array(observations_sim_1)
    observations_sim_2 = np.array(observations_sim_2)

    # next_o = np.concatenate((next_o[0],next_o[1]))
    if len(observations_real.shape) == 1:
        observations_real = np.expand_dims(observations_real, 1)
        observations_sim_1 = np.expand_dims(observations_sim_1, 1)
        observations_sim_2 = np.expand_dims(observations_sim_2, 1)
        # next_o = np.transpose(np.array([next_o[0],next_o[1]]))
        next_o_sim_1 = np.array([next_o[0]])
        next_o_sim_2 = np.array([next_o[1]])
        next_o_real = np.array([next_o[2]])

    next_observations_sim_1 = np.vstack(
        (
            observations_sim_1[1:, :],
            np.expand_dims(next_o_sim_1, 0)
        )
    )
    next_observations_sim_2 = np.vstack(
        (
            observations_sim_2[1:, :],
            np.expand_dims(next_o_sim_2, 0)
        )
    )
    next_observations_real = np.vstack(
        (
            observations_real[1:, :],
            np.expand_dims(next_o_real, 0)
        )
    )
    masks = np.array(masks)

    return dict(
        observations=observations_sim_1,
        actions=actions_sim_1,
        rewards=np.array(rewards_sim_1).reshape(-1, 1),
        next_observations=next_observations_sim_1,
        terminals=np.array(terminals_sim_1).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        masks=masks,
    ), dict(
        observations = observations_sim_2,
        actions=actions_sim_2,
        rewards=np.array(rewards_sim_2).reshape(-1, 1),
        next_observations=next_observations_sim_2,
        terminals=np.array(terminals_sim_2).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        masks=masks,
    ), dict(
        observations = observations_real,
        actions=actions_real,
        rewards=np.array(rewards_real).reshape(-1, 1),
        next_observations=next_observations_real,
        terminals=np.array(terminals_real).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        masks=masks,
    )

# def ensemble_ucb_rollout(
#         env,
#         agent,
#         critic1,
#         critic2,
#         inference_type,
#         feedback_type,
#         num_ensemble,
#         noise_flag=0,
#         max_path_length=np.inf,
#         ber_mean=0.5,
#         render=False,
#         render_kwargs=None,
# ):
#     """
#     The following value for the following keys will be a 2D array, with the
#     first dimension corresponding to the time dimension.
#      - observations
#      - actions
#      - rewards
#      - next_observations
#      - terminals

#     The next two elements will be lists of dictionaries, with the index into
#     the list being the index into the time
#      - agent_infos
#      - env_infos
#     """
#     if render_kwargs is None:
#         render_kwargs = {}
#     observations = []
#     actions = []
#     rewards = []
#     terminals = []
#     agent_infos = []
#     env_infos = []
#     masks = [] # mask for bootstrapping
#     o = env.reset()
#     for en_index in range(num_ensemble):
#         agent[en_index].reset()
#     next_o = None
#     path_length = 0
#     if render:
#         env.render(**render_kwargs)
        
#     while path_length < max_path_length:
#         a_max, ucb_max, agent_info_max = None, None, None
#         for en_index in range(num_ensemble):
#             _a, agent_info = agent[en_index].get_action(o)
#             ucb_score = get_ucb_std(o, _a, inference_type, critic1, critic2,
#                                     feedback_type, en_index, num_ensemble)
            
#             if en_index == 0:
#                 a_max = _a
#                 ucb_max = ucb_score
#                 agent_info_max = agent_info
#             else:
#                 if ucb_score > ucb_max:
#                     ucb_max = ucb_score
#                     a_max = _a
#                     agent_info_max = agent_info

#         next_o, r, d, env_info = env.step(a_max)
#         if noise_flag == 1:
#             r += np.random.normal(0,1,1)[0]
#         observations.append(o)
#         rewards.append(r)
#         terminals.append(d)
#         actions.append(a_max)
#         agent_infos.append(agent_info_max)
#         env_infos.append(env_info)
#         mask = torch.bernoulli(torch.Tensor([ber_mean]*num_ensemble))
#         if mask.sum() == 0:
#             rand_index = np.random.randint(num_ensemble, size=1)
#             mask[rand_index] = 1
#         mask = mask.numpy()
#         masks.append(mask)
  
#         path_length += 1
#         if d:
#             break
#         o = next_o
#         if render:
#             env.render(**render_kwargs)
    
#     actions = np.array(actions)
#     if len(actions.shape) == 1:
#         actions = np.expand_dims(actions, 1)
#     observations = np.array(observations)
#     if len(observations.shape) == 1:
#         observations = np.expand_dims(observations, 1)
#         next_o = np.array([next_o])
#     next_observations = np.vstack(
#         (
#             observations[1:, :],
#             np.expand_dims(next_o, 0)
#         )
#     )
#     masks = np.array(masks)

#     return dict(
#         observations=observations,
#         actions=actions,
#         rewards=np.array(rewards).reshape(-1, 1),
#         next_observations=next_observations,
#         terminals=np.array(terminals).reshape(-1, 1),
#         agent_infos=agent_infos,
#         env_infos=env_infos,
#         masks=masks,
#     )

def ensemble_real_rollout(
        env,
        agent,
        num_ensemble,
        num_step,
        max_path_length=np.inf,
):

    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    for en_index in range(num_ensemble):
        agent[en_index].reset()
    next_o = None
    path_length = 0
    client = CollectionRequest()
    observations, actions = client.request(agent,env,1,1,10)


    # while path_length < max_path_length:
    #     a = None
    #     for en_index in range(num_ensemble):
    #         _a, agent_info = agent[en_index].get_action(o)
    #         if en_index == 0:
    #             a = _a
    #         else:
    #             a += _a
    #     a = a / num_ensemble
    #     next_o, r, d, env_info = env.step(a)
    #     observations.append(o)
    #     rewards.append(r)
    #     terminals.append(d)
    #     actions.append(a)
    #     agent_infos.append(agent_info)
    #     env_infos.append(env_info)
    #     path_length += 1
    #     if d:
    #         break
    #     o = next_o

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
def ensemble_eval(
    env,
    agent,
    num_ensemble,
    max_path_length=10000,
    render=False,
    render_kwargs=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    r_sum = 0
    o = env.reset()
    # for en_index in range(num_ensemble):
    #     agent[en_index].reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a = None
        # for en_index in range(num_ensemble):
        #     _a, agent_info = agent[en_index].get_action(o)
        #     if en_index == 0:
        #         a = _a
        #     else:
        #         a += _a
        # a = a / num_ensemble
        a, agent_info = agent[np.random.randint(0,num_ensemble)].get_action(o)
        next_o, r, d, env_info = env.step(a)
        r_sum += r
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)
    print("avg reward", r_sum/path_length)
    return r_sum/path_length


def ensemble_eval_rollout(
        env,
        agent,
        num_ensemble,
        max_path_length=10000,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    for en_index in range(num_ensemble):
        agent[en_index].reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a = None
        for en_index in range(num_ensemble):
            _a, agent_info = agent[en_index].get_action(o)
            if en_index == 0:
                a = _a
            else:
                a += _a
        a = a / num_ensemble
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    ), mean(rewards)

