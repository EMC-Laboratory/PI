# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:47:11 2020

@author: lingzhang0319
"""

import numpy as np
#from Online_GA_Unchanged import geneticalgorithm as ga
#from modOnlineGAedit import geneticalgorithm as ga
from modOnlineGA import geneticalgorithm as ga
from copy import deepcopy
import ShapePDN as pdn
import matplotlib.pyplot as plt
import time
from config2 import Config
import copy
import math
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

#
def f(X):

    # for future, maybe changing the generated numpy array to always be int so that I don't have to cast the type,
    # i can remore the astype line and change deepcopy --> copy
    z = copy.deepcopy(z2)
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
z2 = opt.input_net.z

# Extract out z from npz file
z_targets = np.load('Supplement MidPts RL R 2 = 6mOhms to 10mOhms 2 Groups Zmax = 20 to 25mOhms 5 groups 5 Per Group.npz')['targets']
z_target = z_targets[0, :]
for i in range(len(z_target)):
    z_target[i] = 0.00075
#R = 0.004
#Zmax = 0.015
#z_target = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log')
print(z_target)
num_loops  = 5
directory_path = 'Compare Different Methods'
conv_save_path = "Compare Methods Old GA Conv 50 Ports Board 2"
sols_save_path = "Compare Methods Old GA Sol 50 Ports Board 2"
conv_save_path = os.path.join(directory_path,conv_save_path)
sols_save_path = os.path.join(directory_path,sols_save_path)

if not os.path.exists(conv_save_path):
    os.makedirs(conv_save_path)
if not os.path.exists(sols_save_path):
    os.makedirs(sols_save_path)

target_num = 1

for i in range(num_loops):
    loop_start_time = time.time()
    # Set GA up
    num_ports = z.shape[1] - 1

    varbound = np.array([[0, 10]] * num_ports)
    algorithm_param = {'max_num_iteration': 50,
                       'population_size': 50,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': None}

    # files to write information
    f_conv_name = "Target Impedance {} Conv {}".format(target_num,i+1)
    f_conv_name = f_conv_name + ".txt"
    f_conv_name = os.path.join(conv_save_path,f_conv_name)
    if os.path.isfile(f_conv_name):
        raise FileExistsError("The file for writing convergence curve scores, {}, already exists".format(f_conv_name))
    else:
        f_conv = open(f_conv_name, "w+")

    record_solutions = True
    f_sols = None
    if record_solutions:
        record_solutions = True
        f_sols_name = "Target Impedance {} Sols {}".format(target_num,i+1)
        f_sols_name = f_sols_name + ".txt"
        f_sols_name = os.path.join(sols_save_path, f_sols_name)
        if os.path.isfile(f_sols_name):
            raise FileExistsError("The file for writing decap solutions, {}, already exists".format(f_sols_name))
        else:
            f_sols = open(f_sols_name, "w+")

    model = ga(function=f,
               dimension=num_ports,
               variable_type='int',
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param,
               seed_sol= None,
               L_mat = None,
               f_sols = f_sols)
    start_time = time.time()
    model.run()
    finish_time = time.time()
    f_conv.write("Time taken =" + str(finish_time - start_time) + "\n")
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
    print('\n Data Collection for Impedance Target {} Complete. Time taken: {}'.format(i,time.time() - loop_start_time))



# total_time = time.time() - start
# np.savez(test_file_name, sols = final_solutions, time = total_time, sol_found = sol_found_array)
# load_found = np.load(test_file_name + ".npz")['sol_found']
# print(load_test)
# print(load_time)
# print(load_found)

