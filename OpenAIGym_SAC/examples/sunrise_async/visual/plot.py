from statistics import variance
from turtle import color
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
import csv

# with open('stat_stable_1.pickle', 'rb') as handle:
#     stat_exp = pickle.load(handle)

# with open('stat_baseline.pickle', 'rb') as handle:
#     stat_baseline = pickle.load(handle)

# with open('stat_stable_onethird_4.pickle', 'rb') as handle:
#     stat_baseline_cf2 = pickle.load(handle)

# with open('stat_stable_onethird_2.pickle', 'rb') as handle:
#     stat_exp_cf2 = pickle.load(handle)

# with open('stat_stable_onethird_3.pickle', 'rb') as handle:
#     stat_exp_cf3 = pickle.load(handle)

# with open('stat_stable_075g.pickle', 'rb') as handle:
#     stat_exp_g9 = pickle.load(handle)

# plt.plot(stat['Log_pi'])
# plt.ylabel("Log pi")

# plt.plot(stat_baseline['Policy_loss'], label='Policy loss')
# plt.plot(stat_exp['Policy_loss'], label='Policy loss')
# plt.plot(stat_exp_cf2['Policy_loss'], label='exp_config2')
# plt.plot(stat_baseline_cf2['Policy_loss'], label='baseline_config2')
# plt.ylabel("Policy loss")

# plt.plot(stat_exp['Critic_loss'], label='Critic loss')
# plt.ylabel("Q value")

# plt.ylim(0.499,0.55)
# plt.plot(stat_baseline['Weight'], label='baseline')
# plt.plot(stat_exp['Weight'], label='exp')
# plt.plot(stat_exp_cf2['Weight'], label='exp_2')
# plt.plot(stat_exp_cf3['Weight'], label='exp_3')
# plt.plot(stat_baseline_cf2['Weight'], label='exp_4')
# plt.ylabel("Weight")
# plt.xlabel("Epoch")

# print(stat_exp['Weight'])

def plot_variance():
    neg_100 = [5.47499294, 0.06814812, 4.31300424]
    neg_200 = [7.01552572, 7.50900475, 2.59096168]
    neg_300 = [8.61452755, 1.18645735, 2.26360121]
    neg_400 = [7.24106782, 9.09057896, 8.5262636]
    neg_500 = [8.36022894, 3.5146496, 5.98741297]
    pos_100 = [2.78918447, 0.1578741]
    pos_200 = [4.11884027, 1.01212394]
    pos_300 = [4.4281675, 7.50948269]
    pos_400 = [8.02480897, 4.18816857]
    pos_500 = [3.34996339, 5.07530692]
    baseline = np.array([0.55327723, 3.41551509, 3.85589404, 2.04910878, 1.48892289])

    neg = [neg_100, neg_200, neg_300, neg_400, neg_500]
    pos = [pos_100, pos_200, pos_300, pos_400, pos_500]

    mean_neg = []
    var_neg = []
    mean_pos = []
    var_pos = []
    for i in range(5):
        p = np.array(neg[i])
        mean_neg.append(np.mean(p))
        var_neg.append(np.var(p))

    for i in range(5):
        p = np.array(pos[i])
        mean_pos.append(np.mean(p))
        var_pos.append(np.var(p))
    
    x = np.array([100,200,300,400,500])
    y_neg = np.array(mean_neg)
    y_pos = np.array(mean_pos)
    variance_neg = np.array(var_neg)
    variance_pos = np.array(var_pos)

    plt.plot(x, y_neg, label='neg')
    plt.plot(x, y_pos, label='pos')
    plt.plot(x, baseline, label='baseline')
    # plt.fill_between(x, y_neg-variance_neg, y_neg+variance_neg)
    # plt.fill_between(x, y_pos-variance_pos, y_pos+variance_pos)
    plt.legend()
    plt.show()

def plot_reward():
    num_slices = 10
    slice_len = 1
    # baseline = [np.mean(stat_baseline['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
    exp = [np.mean(stat_exp['R_sum'][i:i+slice_len]) for i in range(0,1000,slice_len)]
    # baseline_cf2 = [np.mean(stat_baseline_cf2['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
    # exp_cf2 = [np.mean(stat_exp_cf2['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
    # exp_cf3 = [np.mean(stat_exp_cf3['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]
    # exp_g9 = [np.mean(stat_exp_g9['R_sum'][i:i+slice_len]) for i in range(0,200,slice_len)]

    # plt.plot(exp_cf2, label='exp2', color='orange')
    # plt.plot(baseline_cf2, label='0.75g')
    # plt.plot(baseline, label='baseline')
    plt.plot(exp, label='exp', color='blue')
    # plt.plot(exp_cf3, label='exp3', color='green')
    # plt.plot(baseline_cf2, label='exp4', color='red')
    # plt.plot(exp_g9, label='0.75g')
    # plt.ylabel("Reward")

    plt.legend()
    plt.show()

def plot_distribution():
    with open('simreal_neg.csv', mode='r') as handle:
        reader = csv.reader(handle)
        rewards_neg = []
        for row in reader:
            for value in row:
                rewards_neg.append(float(value))

    with open('simreal_baseline.csv', mode='r') as handle:
        reader = csv.reader(handle)
        rewards_baseline = []
        for row in reader:
            for value in row:
                rewards_baseline.append(float(value))
                

    bins = np.linspace(-30, 30, 50)

    plt.hist(rewards_neg, bins, alpha=0.5, label='x')
    plt.hist(rewards_baseline, bins, alpha=0.5, label='y')
    plt.legend(loc='upper right')
    plt.show()
    # print(rewards)

plot_distribution()