import pickle


with open("stat.pickle", 'rb') as handle:
    stat = pickle.load(handle)
    print(stat['QF1 Loss'])
    print(stat['QF2 Loss'])