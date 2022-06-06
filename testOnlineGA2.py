# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:47:11 2020

@author: lingzhang0319
"""

import numpy as np
#from Online_GA_Unchanged import geneticalgorithm as ga
from modOnlineGA2 import geneticalgorithm as ga
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


BASE_PATH = 'new_data_test/'
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

L = np.load(os.path.join('Testing for Moving Vias', '100 Port Boundary Mat Test CCW' +'.npz'))['L']


# Extract out the self L and mutual L of between/of power pins only
num_IC = 21
num_pwr = 9
num_decap_ports = 101
num_vias = num_IC + num_decap_ports * 2

# delete out ground pins
del_ports = [9 + i for i in range(num_IC - num_pwr)] + [ (num_IC-1) + 2*j for j in range(1,num_decap_ports+1)]
del_ports.reverse()
for i in del_ports:
    L = np.delete(np.delete(L,i,1),i,0)

# delete out power pin of VRM via
L = np.delete(np.delete(L,L.shape[0]-1,1), L.shape[0]-1,0)

# Merge IC power pins
L = merge_ports(L,[0,1,2,3,4,5,6,7,8])


def f(X):
    z_in = deepcopy(z)
    decap_map = deepcopy(X)
    decap_map = decap_map.astype(int)

    map_z = pdn.new_connect_n_decap(z_in, decap_map, cap_objs_z, opt)
    z_solution = np.abs(map_z)

    if np.count_nonzero(np.greater(z_solution, z_target)) == 0: # <---- If target met

        reward = len(decap_map) - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports

        #last_z_mod = 1 + z_solution[-1]/z_target[-1]               # <----- How close solution is to tarrget
        #reward = reward * last_z_mod

    else:
        reward = -np.max((z_solution - z_target) / z_target)           # <--- decrease largest z pt
        #reward = reward + -1 * (abs(z_solution[-1]-z_target[-1]))      # <--- how close to last target pt


    return -reward

def OptionsInit():
    # Get settings
    opt = Config()
    return opt




def decap_objects(opt):
    # Store capacitor library as their impedances
    cap_objs = [pdn.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors, 1 to 10, high end is not inclusive [x,y) so need + 1
    cap_objs_z = deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z


###### Set settings ######
opt = OptionsInit()
_,cap_objs_z = decap_objects(opt)


#Set target

# Used for test board
# 5000.s21-
# R = .01
# Zmax = .015


# Used for test board
# 10200.s21-
# R = .01
# Zmax = .02

#100 cap CW
R = .011
Zmax = .021

# 100 Cap ccw
R = .012
Zmax = .026

# 50 Cap
# R = .01
# Zmax = .024
R = 0.034
Z = 0.034
z_target = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=opt.nf, interp='log')

z = opt.input_net.z


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
# seed_sol = np.array([ 0,  2,  1 , 6 , 3,  0,  4,  1,  0,  1,  1,  0,  0,  3,  6,  0,  2,  1,  2,  0,  1,  9,  0,  2,
# 0 , 1 , 1,  0,  0,  0,  9,  0,  0,  1,  0,  0,  3,  0,  1,  3, 10,  5 , 1 , 0 , 3 , 2,  1,  0,
#    1,  1,])
seed_sol = None

#modded onesc
model = ga(function=f,
           dimension=num_ports,
           variable_type='int',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param,
           seed_sol= None,
           L_mat = L)


# file to write information
f = open("100 Cap Mod, Shorted Pop Control and Shorted Sort.txt", "a")


t1 = time.time()
model.run()
f.write("Time taken =" + str(time.time() - t1) + "\n")

print(time.time() - t1)

convergence = model.report

decap_solution = model.output_dict['variable']
decap_solution = decap_solution.astype(int)
best_z = pdn.new_connect_n_decap(z, decap_solution, cap_objs_z, opt)

## Final Check ##
# Check if the min decap and best map sol can have their decap # decreased (Primitively)

#Check if best solution can be improved
#decap_solution = final_check(decap_solution,opt,cap_objs_z)

# Recalculate Impedance, whether it improves or not
#best_z = pdn.new_connect_n_decap(z, decap_solution, cap_objs_z, opt)




# Print Final Results
print('Best solution is:', decap_solution.tolist())

# Write things of interest to file for data collection

f.write('Decap Solution. Num caps =' +  str(np.count_nonzero(decap_solution)) + "\n")
f.write(str(decap_solution.tolist()))
f.write('\n')
f.write('Convergence Curve (Scores)' + '\n')
f.write(str(convergence))
f.write("\n,\n")
f.close()

plt.loglog(opt.freq, z_target, '--', color= 'black')
plt.loglog(opt.freq, np.abs(best_z))
plt.title('Best Solution',fontsize = 16)
plt.xlabel('Freq in Hz',fontsize = 16)
plt.ylabel('Impedance in Ohms', fontsize = 16)
plt.legend(['Target Z', 'Best Solution'],prop={'size': 12})
plt.grid(True, which = 'Both')
#plt.savefig("Final_Test_100_Gen_2.png")
plt.show()

