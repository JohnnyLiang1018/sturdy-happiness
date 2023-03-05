import matplotlib.pyplot as plt
import pickle

with open('stat_hpc3.pickle', 'rb') as handle:
    stat = pickle.load(handle)

plt.plot(stat['Policy_loss'])
plt.show()