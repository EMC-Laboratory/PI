import numpy as np
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


def get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log'):
    f_transit = fstop * R / Zmax
    if interp == 'log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return ztarget

def OptionsInit():
    # Get settings
    opt = Config()
    return opt

def decap_objects(opt):
    # Store capacitor library as their impedances
    cap_objs = [pdn.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors, 1 to 10, high end is not inclusive [x,y) so need + 1
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z

###### Set settings ######
opt = OptionsInit()
_,cap_objs_z = decap_objects(opt)
z = opt.input_net.z


#
# input_file = np.load('NN Input 500 Inputs For Distributions.npz')
# targets = input_file['ztargets']
# solution_maps = input_file['sols']
# inputs = input_file['inputs']
# cap_labels = input_file['labels']
# distribution = input_file['dist']
#
# no_target_met = []
# print(np.shape(targets))
# print(np.shape(cap_labels))
# print(np.shape(distribution))
# print(np.shape(inputs))
# print(np.shape(solution_maps))
# for i in range(500):
#     z1 = pdn.new_connect_n_decap(z, solution_maps[i], cap_objs_z, opt)
#     if np.count_nonzero(np.greater(np.abs(z1), targets[i])) != 0:
#         print(i)
#         no_target_met.append(i)
# print(500 - len(no_target_met))
# targets = np.delete(targets, no_target_met, axis= 0)
# cap_labels = np.delete(cap_labels, no_target_met, axis = 0)
# distribution = np.delete(distribution, no_target_met, axis = 0)
# inputs = np.delete(inputs, no_target_met, axis = 1)
# solution_maps = np.delete(solution_maps, no_target_met, axis = 0)
#
# print(np.shape(targets))
# print(np.shape(cap_labels))
# print(np.shape(distribution))
# print(np.shape(inputs))
# print(np.shape(solution_maps))
# print(cap_labels)
#
# z1 = pdn.new_connect_n_decap(z, solution_maps[40], cap_objs_z, opt)
# z2 = pdn.new_connect_n_decap(z, solution_maps[41], cap_objs_z, opt)
# t1 = targets[40]
# t2 = targets[41]
# plt.loglog(opt.freq, abs(z1))
# plt.loglog(opt.freq, t1)
# plt.show()

file_name = "NN 500 Inputs, Target Met Only"
#np.savez(file_name, inputs = inputs, labels = cap_labels, ztargets = targets, dist = distribution, sols = solution_maps)

#input_file = np.load('Supplement MidPts RL R 2 = 6mOhms to 10mOhms 2 Groups Zmax = 20 to 25mOhms 5 groups 5 Per Group.npz')
#label_file = np.load('Middling Caps Test.npz')
#solution_maps_file = np.load("Final Check with Middling Caps.npz")

input_file = np.load("NN 500 Inputs, Target Met Only.npz")
R = input_file['R_Ohms']
Z = input_file['Zmax_Ohms']
L = input_file['L_Henries']
f = input_file['trans_f_Hz']
targets = input_file["targets"]
solution_maps = input_file['sols']
z1 = pdn.new_connect_n_decap(z, solution_maps[40], cap_objs_z, opt)
z2 = pdn.new_connect_n_decap(z, solution_maps[41], cap_objs_z, opt)
t1 = targets[40]
t2 = targets[41]
plt.loglog(opt.freq, abs(z1))
plt.loglog(opt.freq, t1)
plt.show()