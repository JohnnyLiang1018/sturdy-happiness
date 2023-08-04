from turtle import color
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

with open('stat_stable_1.pickle', 'rb') as handle:
    stat_exp = pickle.load(handle)

with open('stat_baseline.pickle', 'rb') as handle:
    stat_baseline = pickle.load(handle)

with open('stat_stable_onethird_4.pickle', 'rb') as handle:
    stat_baseline_cf2 = pickle.load(handle)

with open('stat_stable_onethird_2.pickle', 'rb') as handle:
    stat_exp_cf2 = pickle.load(handle)

with open('stat_stable_onethird_3.pickle', 'rb') as handle:
    stat_exp_cf3 = pickle.load(handle)

with open('stat_stable_075g.pickle', 'rb') as handle:
    stat_exp_g9 = pickle.load(handle)

# plt.plot(stat['Log_pi'])
# plt.ylabel("Log pi")

# plt.plot(stat_baseline['Policy_loss'], label='Policy loss')
plt.plot(stat_exp['Policy_loss'], label='Policy loss')
# plt.plot(stat_exp_cf2['Policy_loss'], label='exp_config2')
# plt.plot(stat_baseline_cf2['Policy_loss'], label='baseline_config2')
# plt.ylabel("Policy loss")

# plt.plot(stat_exp['Critic_loss'], label='Critic loss')
# plt.ylabel("Q value")

num_slices = 10
slice_len = 3
# baseline = [np.mean(stat_baseline['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# exp = [np.mean(stat_exp['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# baseline_cf2 = [np.mean(stat_baseline_cf2['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# exp_cf2 = [np.mean(stat_exp_cf2['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# exp_cf3 = [np.mean(stat_exp_cf3['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# exp_g9 = [np.mean(stat_exp_g9['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]

# plt.plot(exp_cf2, label='exp2', color='orange')
# plt.plot(baseline_cf2, label='0.75g')
# plt.plot(baseline, label='baseline')
# plt.plot(exp, label='exp', color='blue')
# plt.plot(exp_cf3, label='exp3', color='green')
# plt.plot(baseline_cf2, label='exp4', color='red')
# plt.plot(exp_g9, label='0.75g')
# plt.ylabel("Reward")

# plt.ylim(0.499,0.55)
# plt.plot(stat_baseline['Weight'], label='baseline')
# plt.plot(stat_exp['Weight'], label='exp')
# plt.plot(stat_exp_cf2['Weight'], label='exp_2')
# plt.plot(stat_exp_cf3['Weight'], label='exp_3')
# plt.plot(stat_baseline_cf2['Weight'], label='exp_4')
# plt.ylabel("Weight")
# plt.xlabel("Epoch")

plt.legend()
plt.show()

# print(stat_exp['Weight'])