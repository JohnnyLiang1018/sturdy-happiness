import matplotlib.pyplot as plt
import torch
import pickle

with open('stat_exp3_validate.pickle', 'rb') as handle:
    stat = pickle.load(handle)
    # stat = torch.load(handle,map_location=torch.device('cpu'))

# plt.plot(stat['Log_pi'])
# plt.ylabel("Log pi")

plt.plot(stat['Policy_loss'])
plt.ylabel("Policy loss")

# plt.plot(stat['Q_action'])
# plt.ylabel("Q value")

# plt.plot(stat['R_sum'])
# plt.ylabel("Reward")

# plt.ylim(0.49999,0.50005)
# plt.plot(stat['Weight'])
# plt.ylabel("Weight")

plt.xlabel("Epoch")
plt.show()

# print(stat['Weight'][:99])