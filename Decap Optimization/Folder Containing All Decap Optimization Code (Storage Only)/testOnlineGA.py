# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:47:11 2020

@author: lingzhang0319
"""

import numpy as np
from Online_GA_Unchanged import geneticalgorithm as ga
#from modOnlineGAedit import geneticalgorithm as ga
#from modOnlineGA import geneticalgorithm as ga
from copy import deepcopy
import ShapePDN as pdn
import matplotlib.pyplot as plt
import time
from config2 import Config
import copy
import os

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


def merge_ports(L_orig, merge_port_list, kept_port=0):
    # merge_port_list, kept_port index begins with 0

    orig_port_list = list(range(0, L_orig.shape[0])) # get the number of ports
    del_ports = deepcopy(merge_port_list)            # delete out the ports to be merged
    del_ports.remove(kept_port)                      # get the port number we want to merge the ports to

    # get the remaining ports (ports that are not to be merged
    left_port_list = deepcopy(orig_port_list)
    for a in del_ports:
        left_port_list.remove(a)

    # get inverse L matrix
    l_inv = np.linalg.inv(L_orig)
    if len(orig_port_list) > len(merge_port_list):
        # merge ports by adding the corresponding rows and columns
        reduce_list = list([kept_port,merge_port_list[-1]+1]) + list(range(merge_port_list[-1]+2,orig_port_list[-1]+1))
        l_inv_merge = np.add.reduceat(np.add.reduceat(l_inv, reduce_list,axis=0),reduce_list,axis=1)
    else:
        l_inv_merge = np.add.reduceat(np.add.reduceat(l_inv, [0],axis=0),[0],axis=1)
    l_merge = np.linalg.inv(l_inv_merge)
    # port_map_orig is the port map to the original port number

    return l_merge


BASE_PATH = 'new_data_test/'
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

#L = np.load(os.path.join('Testing for Moving Vias', '100 Port Boundary Mat Test CCW' +'.npz'))['L']
# # Extract out the self L and mutual L of between/of power pins only
# num_IC = 21
# num_pwr = 9
# num_decap_ports = 101
# num_vias = num_IC + num_decap_ports * 2
#
# # delete out ground pins
# del_ports = [9 + i for i in range(num_IC - num_pwr)] + [ (num_IC-1) + 2*j for j in range(1,num_decap_ports+1)]
# del_ports.reverse()
# for i in del_ports:
#     L = np.delete(np.delete(L,i,1),i,0)
#
# # delete out power pin of VRM via
# L = np.delete(np.delete(L,L.shape[0]-1,1), L.shape[0]-1,0)
#
# # Merge IC power pins
# L = merge_ports(L,[0,1,2,3,4,5,6,7,8])
#

#
def f(X):

    # for future, maybe changing the generated numpy array to always be int so that I don't have to cast the type,
    # i can remore the astype line and change deepcopy --> copy

    #z = copy.deepcopy(z2)
    z = copy.deepcopy(opt.input_net.z)
    decap_map = copy.copy(X)
    decap_map = decap_map.astype(int)

    map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
    z_solution = np.abs(map_z)

    if np.count_nonzero(np.greater(z_solution, z_target)) == 0: # <---- If target met

        reward = len(decap_map) - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports
        #reward = math.log(reward,10)
        #last_z_mod = 1 + z_solution[-1]/z_target[-1]               # <----- How close solution is to tarrget
        #reward = reward * last_z_mod

    else:
        reward = -np.max((z_solution - z_target) / z_target)           # <--- decrease largest z pt
        # reward = math.inf
        # for i in range(len(z_solution)):
        #     temp_reward = (z_solution[i] - z_target[i])/z_target[i]
        #     reward = temp_reward if (temp_reward < reward and temp_reward > 0) else reward
        # reward = -reward
    return -reward
#
# def f(X): #paulise scoring
#
#     decap_map = deepcopy(X)
#     decap_map = decap_map.astype(int)
#
#     z_solution = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
#     abs_map_z = np.abs(copy.deepcopy(z_solution))
#     score_holder = 0
#
#     if np.count_nonzero(np.greater(abs_map_z, z_target)) == 0: # <---- If target met
#         reward = len(decap_map) - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports
#
#     else:
#         for index in range(len(abs_map_z)):
#             add_score = abs(abs_map_z[index] - z_target[index]) if abs(abs_map_z[index]) >= abs(opt.ztarget[index]) else 0
#             score_holder = score_holder + add_score
#         pts_above = np.count_nonzero(np.greater(abs_map_z, opt.ztarget))
#         pts_above = pts_above if pts_above > 0 else 1
#         reward = -score_holder / pts_above
#
#     return -reward

# def f(X, zeros_prev = None):
#
#     z = deepcopy(opt.input_net.z)
#     decap_map = deepcopy(X)
#     decap_map = decap_map.astype(int)
#     map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
#     z_solution = np.abs(map_z)
#
#     if np.count_nonzero(np.greater(z_solution, z_target)) == 0: # <---- If target met
#
#         reward = len(decap_map) - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports
#
#         if zeros_prev is not None:
#             reward = reward - zeros_prev
#
#     else:
#         reward = -np.max((z_solution - z_target) / z_target)           # <--- decrease largest z pt
#         #reward = reward + -1 * (abs(z_solution[-1]-z_target[-1]))      # <--- how close to last target pt
#
#
#     return -reward



# def f(X):
#     z = deepcopy(opt.input_net.z)
#
#     decap_map = deepcopy(X)
#
#     decap_map = decap_map.astype(int)
#
#     map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
#     z_solution = np.abs(map_z)
#
#     if np.count_nonzero(np.greater(z_solution, z_target)) == 0:
#         reward = len(z_target)
#         mod = (1 - abs(np.min((z_target - z_solution) / z_target))) if (1 - abs(np.min((z_target - z_solution) / z_target))) > 0 else 1
#         mod_ind = np.argmin((z_target - z_solution) / z_target)/len(z_target)
#
#         mod2 = np.min(z_solution) / z_target[np.argmin(z_solution)]
#
#         mod2 = np.average(z_solution)/np.average(z_target)
#         mod3 = 1 - sum(z_target - z_solution)/sum(z_target)
#
#         multiplier = mod*mod_ind + mod3
#         reward = reward * multiplier + len(decap_map) - np.count_nonzero(decap_map) + 1
#     else:
#         mod = (1 - np.max((z_solution - z_target) / z_target)) if (1 - np.max((z_solution - z_target) / z_target)) > 0 else 1
#         mod_ind = np.argmin((z_solution - z_target) / z_target)/len(z_target)
#         multiplier = mod
#         reward = np.count_nonzero(np.less(z_solution, z_target)) * multiplier
#         #print(reward)
#     return -reward
#


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


#Set target

# 50 Cap Test 1
R = 0.01
Zmax = 0.024

## 50 Cap Test 2
# R = .012
# Zmax = 0.0525

## 50 Cap Test 3
# R = 0.009
# Zmax = 0.022

z = opt.input_net.z
z2 = copy.deepcopy(z)


z_target = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log')
print(z_target)

# Set GA up
num_ports = z.shape[1] - 1
varbound = np.array([[0, 10]] * num_ports)
algorithm_param = {'max_num_iteration': 100,
                   'population_size': 100,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': None}
# seed_sol = np.array([ 0,  2,  1 , 6 , 3,  0,  4,  1,  0,  1,  1,  0,  0,  3,  6,  0,  2,  1,  2,  0,  1,  9,  0,  2,
# 0 , 1 , 1,  0,  0,  0,  9,  0,  0,  1,  0,  0,  3,  0,  1,  3, 10,  5 , 1 , 0 , 3 , 2,  1,  0,
#    1,  1,])
seed_sol = None

# files to write information

f_conv_name = "Temp111"
f_conv_name = f_conv_name + ".txt"
if os.path.isfile(f_conv_name):
    raise FileExistsError("The file for writing convergence curve scores, {}, already exists".format(f_conv_name))
else:
    f_conv = open(f_conv_name, "w+")

record_solutions = False
f_sols = None
if record_solutions:
    record_solutions = True
    f_sols_name = "Temp222"
    f_sols_name = f_sols_name + ".txt"
    if os.path.isfile(f_sols_name):
        raise FileExistsError("The file for writing decap solutions, {}, already exists".format(f_sols_name))
    else:
        f_sols = open(f_sols_name, "w+")

# model = ga(function=f,
#            dimension=num_ports,
#            variable_type='int',
#            variable_boundaries=varbound,
#            algorithm_parameters=algorithm_param,
#            seed_sol= None,
#            L_mat = L,
#            f_sols = f_sols)

##normal one
model = ga(function=f,
           dimension=num_ports,
           variable_type='int',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)


t1 = time.time()
model.run()
f_conv.write("Time taken =" + str(time.time() - t1) + "\n")

print(time.time() - t1)

convergence = model.report

decap_solution = model.output_dict['variable']
decap_solution = decap_solution.astype(int)


if 'meetZ' in model.output_dict:
    sol_meeting_target = model.output_dict['meetZ']
else:
    sol_meeting_target = None

if sol_meeting_target is None:
    sol_meeting_target = np.copy(decap_solution)
else:
    sol_meeting_target = sol_meeting_target.astype(int)

# need to check if sol_meeting_target satsifies target
final_z = pdn.new_connect_n_decap(z, decap_solution, cap_objs_z, opt)
best_z = pdn.new_connect_n_decap(z, sol_meeting_target, cap_objs_z, opt)

print('Best Solution in Final GA Population:', decap_solution.tolist())
print('Best Overall Solution Meeting Target:', sol_meeting_target.tolist())

# Write things of interest to file for data collection
f_conv.write('Best Solution in Final GA Population. Num caps =' +  str(np.count_nonzero(decap_solution)) + "\n")
target_met_bool = 'True' if np.count_nonzero(np.greater(abs(final_z), z_target)) == 0 else 'False'
f_conv.write('Target Met? ' + target_met_bool + '\n')
f_conv.write(str(decap_solution.tolist()))
f_conv.write('\n')
f_conv.write('Best Overall Solution meeting Target Impedance. Num caps =' +  str(np.count_nonzero(sol_meeting_target)) + "\n")
f_conv.write(str(sol_meeting_target.tolist()))
f_conv.write('\n')
f_conv.write('Convergence Curve (Scores)' + '\n')
f_conv.write(str(convergence))
f_conv.write("\n,\n")

f_conv.close()
if record_solutions:
    f_sols.close()

# plt.loglog(opt.freq, z_target, '--', color= 'black')
# plt.loglog(opt.freq, np.abs(final_z))
# plt.loglog(opt.freq, np.abs(best_z))
# plt.title('Final Solutions',fontsize = 16)
# plt.xlabel('Freq in Hz',fontsize = 16)
# plt.ylabel('Impedance in Ohms', fontsize = 16)
# plt.legend(['Target Z', 'Best Solution in Final Population', 'Best Solution Satisfying Target Z'],prop={'size': 12})
# plt.grid(True, which = 'Both')
# #plt.savefig("Final_Test_100_Gen_2.png")
# plt.show()



## Final Check ##
# Check if the min decap and best map sol can have their decap # decreased (Primitively)

#Check if best solution can be improved

#decap_solution = final_check(decap_solution,opt,cap_objs_z)

# Recalculate Impedance, whether it improves or not
#best_z = pdn.new_connect_n_decap(z, decap_solution, cap_objs_z, opt)




#
# [[ 1.          2.          0.          0.          1.          6.
#    1.          4.          0.          1.          9.          2.
#    0.          0.         -6.        ]
#  [ 1.          2.          0.          0.          1.          8.
#    1.          4.          0.          1.          9.          2.
#    0.          0.         -6.        ]
#  [ 1.          2.          0.          0.          1.          8.
#    1.          0.          0.          1.          9.          2.
#    0.          0.          0.42085451]
#  [ 1.          2.          0.          0.          1.          8.
#    1.          4.          0.          1.          9.          2.
#    0.          0.         -6.        ]]

# [[ 6.          2.          4.          0.          0.          2.
#    2.          0.          2.          3.          4.          0.
#    2.          9.          0.02773493]
#  [ 8.          7.          3.          0.          5.          2.
#    0.          0.          1.          5.          4.          0.
#    0.          0.          0.25420274]
#  [ 6.          6.          4.         10.          0.          2.
#    0.          0.          0.          0.          4.          3.
#    2.          0.          0.20508636]
#  [ 6.          8.          4.         10.          3.          0.
#    2.          0.          0.          3.          4.          5.
#    0.          9.          0.17084777]
#  [ 6.          2.          4.          0.          0.          0.
#    0.          0.          2.          3.          4.          0.
#    0.          9.          0.26473696]
#  [ 6.          2.          4.          0.          0.          2.
#    2.          0.          2.          3.          4.          0.
#    2.          9.          0.02773493]
#  [ 8.          7.          0.          4.          0.          2.
#    0.          0.          3.          0.          5.          0.
#    0.          9.          0.26347079]
#  [ 6.          2.          4.          0.          0.          2.
#    2.         10.          0.          6.          1.          0.
#    0.          0.          0.18666219]]

