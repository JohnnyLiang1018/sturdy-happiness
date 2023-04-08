import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

with open('stat_exp5_config3.pickle', 'rb') as handle:
    stat_exp = pickle.load(handle)
    # stat = torch.load(handle,map_location=torch.device('cpu'))

with open('stat_stable.pickle', 'rb') as handle:
    stat_baseline = pickle.load(handle)

with open('stat_baseline_config2.pickle', 'rb') as handle:
    stat_baseline_cf2 = pickle.load(handle)

with open('stat_exp4_config2.pickle', 'rb') as handle:
    stat_exp_cf2 = pickle.load(handle)

# plt.plot(stat['Log_pi'])
# plt.ylabel("Log pi")

plt.plot(stat_baseline['Policy_loss'], label='baseline')
plt.plot(stat_exp['Policy_loss'], label='exp')
# plt.plot(stat_exp_cf2['Policy_loss'], label='exp_config2')
# plt.plot(stat_baseline_cf2['Policy_loss'], label='baseline_config2')
plt.ylabel("Policy loss")

# plt.plot(stat['Q_action'])
# plt.ylabel("Q value")

# num_slices = 20
# slice_len = 10
# baseline = [np.mean(stat_baseline['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# baseline_cf2 = [np.mean(stat_baseline_cf2['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# exp = [np.mean(stat_exp['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# exp_cf2 = [np.mean(stat_exp_cf2['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
# plt.plot(baseline, label='baseline')
# plt.plot(exp, label='exp')
# plt.plot(exp_cf2, label='exp_config2')
# plt.plot(baseline_cf2, label='baseline_config2')
# plt.ylabel("Reward")

# plt.ylim(0,0.002)
# plt.plot(stat_baseline['Weight'], label='baseline')
# plt.plot(stat_exp['Weight'], label='exp')
# plt.plot(stat_exp_cf2['Weight'], label='exp_config2')
# plt.plot(stat_baseline_cf2['Weight'], label='baseline_config2')
# plt.ylabel("Weight")

plt.xlabel("Epoch")
plt.legend()
plt.show()

# print(stat_exp['Weight'])