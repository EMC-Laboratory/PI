from config2 import Config
import ShapePDN as pdn
import copy
import PopInit as Pop
import matplotlib.pyplot as plt
import numpy as np
import random
import skrf as rf
import math as math
import sys


file = np.load("Final Check with Middling Caps.npz")
file2 = np.load("Supplement MidPts RL R 2 = 6mOhms to 10mOhms 2 Groups Zmax = 20 to 25mOhms 5 groups 5 Per Group.npz")
tar = file2['targets']
R = file2['R_Ohms']
Z = file2['Zmax_Ohms']
L = file2['L_Henries']
f = file2['trans_f_Hz']
# tar = load_test['targets'] R = load_test['R_Ohms'] Z = load_test['Zmax_Ohms'] L = load_test['L_Henries'] f = load_test['trans_f_Hz']

solution_maps = file["sols"]
distribution = np.ndarray((np.shape(solution_maps)[0],3))
num_cap_array = np.ndarray((np.shape(solution_maps)[0],1))
# Range 1 = caps 1 2 3
# Range 2 = caps 4 5 6 7 8
# range 3 = caps 9 10
for i in range(np.shape(solution_maps)[0]):
    num_caps = np.count_nonzero(solution_maps[i])
    range_1 = np.count_nonzero(solution_maps[i][(solution_maps[i] >= 1) & (solution_maps[i] <= 3)])
    range_2 = np.count_nonzero(solution_maps[i][(solution_maps[i] >= 4) & (solution_maps[i] <= 8)])
    range_3 = np.count_nonzero(solution_maps[i][(solution_maps[i] >= 9) & (solution_maps[i] <= 10)])
    distribution[i, 0] = range_1  /num_caps
    distribution[i, 1] = range_2 / num_caps
    distribution[i, 2] = range_3 / num_caps
    num_cap_array[i] = num_caps
    print(num_caps)


file3 = np.load('NN Input.npz')
targets = file3['ztargets']
for i in range(500):
    plt.loglog(opt.freq, targets[i])

plt.xlabel('Freq')
plt.ylabel('Target Z')
plt.title('Target Impedances')
plt.grid(which = 'both')
plt.show()
#
# plt.scatter(range(np.shape(solution_maps)[0]), distribution[:,0])
# plt.scatter(range(np.shape(solution_maps)[0]), distribution[:,1])
# plt.scatter(range(np.shape(solution_maps)[0]), distribution[:,2])
# plt.legend(['Group 1', 'Group 2','Group 3'])
# plt.xlabel('Impedance Target')
# plt.ylabel('% in Best Solution')
# plt.title('Distribution of Capacitor Groups in Generated Target Impedances')
# plt.show()
# test_nn_file = "Middling Caps Test"
# #holder = np.savez(test_nn_file, dists = distribution, z = tar, num_caps = num_cap_array)
