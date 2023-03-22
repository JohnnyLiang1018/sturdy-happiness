import matplotlib.pyplot as plt
import torch
import pickle

with open('stat_stable.pickle', 'rb') as handle:
    stat = pickle.load(handle)
    # stat = torch.load(handle,map_location=torch.device('cpu'))

plt.plot(stat['Weight'])
plt.show()

# print(len(stat['Weight_actor_q']))