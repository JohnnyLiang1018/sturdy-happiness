import matplotlib.pyplot as plt
import torch
import pickle

with open('stat_stable3.pickle', 'rb') as handle:
    stat = pickle.load(handle)
    # stat = torch.load(handle,map_location=torch.device('cpu'))

plt.plot(stat['Policy_loss'])
plt.show()

# print(len(stat['Weight_actor_q']))