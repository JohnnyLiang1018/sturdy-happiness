import matplotlib.pyplot as plt
import pickle

with open('stat_cross_std.pickle', 'rb') as handle:
    stat = pickle.load(handle)

plt.plot(stat['Policy_loss'])
plt.show()

# print(len(stat['Weight_actor_q']))