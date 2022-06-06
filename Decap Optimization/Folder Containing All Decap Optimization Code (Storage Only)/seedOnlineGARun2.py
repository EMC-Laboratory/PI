
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:47:11 2020

@author: lingzhang0319
"""

import numpy as np
#from Online_GA_Unchanged import geneticalgorithm as ga
from seedOnlineGA2 import geneticalgorithm as ga
from copy import deepcopy
import ShapePDN as pdn
import matplotlib.pyplot as plt
import time
from config2 import Config
import copy


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

# def target_shift_index(z_target):
#
#     # Could pull this off the text file directly but it works
#     # also definitely much
#
#     shift_ind = []
#
#
#     if np.amax(z_target) == np.amin(z_target):
#         # constant curve, zero slope
#         print('Target Curve is Constant, target shift point set as last point')
#         shift_ind = np.shape(z_target)[0]
#
#     else:
#         for index in range(opt.nf - 1):
#             if z_target[index + 1] - z_target[index] != 0:
#                 shift_ind = index
#                 print('Point where slope changes occurs at index', index, 'with f =', opt.freq[index])
#                 break
#     return shift_ind

def f(X):


    z = deepcopy(opt.input_net.z)

    decap_map = deepcopy(X)

    decap_map = decap_map.astype(int)

    map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
    z_solution = np.abs(map_z)

    if np.count_nonzero(np.greater(z_solution, z_target)) == 0: # <---- If target met

        reward = num_ports - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports

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



def final_check(min_zero_map,opt,cap_objs_z): # should use this at some point
    improve_bool = True
    while improve_bool is True:
        current_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # number of caps in the initial min_zero_map
        print('Before check, decap map is:', min_zero_map)
        for ind, _ in enumerate(min_zero_map): # iterating through each decap in min map
            holder = copy.deepcopy(min_zero_map) # make copy

            if min_zero_map[ind] != 0: # if port is not empty
                holder[ind] = 0  # make port empty
                holder_z = pdn.new_connect_n_decap(opt.input_net.z, holder, cap_objs_z, opt)
                if np.count_nonzero(np.greater(np.absolute(holder_z), opt.ztarget)) == 0:
                    # if # of capacitors decrease and target met, overwrite min zero map
                    min_zero_map = copy.deepcopy(holder) # update to better map
                    # improve_bool still true
                    break
                else:
                    holder = copy.deepcopy(min_zero_map)
                    # if target impedance not met, recapture min_zero_map
                    # and set the next non-empty port 0
        new_min = len(min_zero_map) - np.count_nonzero(min_zero_map) # used to set improve bool
        if new_min > current_min:
            print('After check, number of capacitors decreased. Min decap layout is', min_zero_map)
            print('Checking Again....')
            improve_bool = True # not needed but helps me with clarity
        else:
            print('After check, number of capacitors did not decrease.')
            improve_bool = False # score did not improve, set improve_bool to false. break out of loop
    return min_zero_map




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

R = .01
Zmax = .024


z_target = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=opt.nf, interp='log')
#z_target = opt.ztarget
#opt.ztarget = z_target
z = opt.input_net.z

# Set objective functions variables
#shift_index = target_shift_index(z_target)

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
seed_sol = np.array([6, 4, 3, 9, 3, 1, 1, 0, 1, 7, 0, 1, 4, 2, 0, 8, 1, 8, 2, 0, 0, 3, 1, 2, 3, 0, 2, 0, 6, 0, 0, 0, 0, 4, 1, 2, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 8, 0, 1, 1])
#seed_sol = np.array([10, 8, 0, 3, 3, 5, 3, 2, 2, 7, 3, 2, 2, 5, 2, 2, 3, 4, 5, 4])

f1 = open("20 Cap No Shift Test 3.txt", "a")
filename = "20 Cap No Shift Test 3.txt"
#modded one
model = ga(function=f,
           dimension=num_ports,
           variable_type='int',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param,
           seed_sol= seed_sol,
           file = filename)

#normal one
# model = ga(function=f,
#            dimension=num_ports,
#            variable_type='int',
#            variable_boundaries=varbound,
#            algorithm_parameters=algorithm_param)


# file to write information


t1 = time.time()
model.run()
f1.write("Time taken =" + str(time.time() - t1) + "\n")

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

f1.write('Decap Solution. Num caps =' +  str(np.count_nonzero(decap_solution)) + "\n")
f1.write(str(decap_solution.tolist()))
f1.write('\n')
f1.write('Convergence Curve (Scores)' + '\n')
f1.write(str(convergence))
f1.write("\n,\n")
f1.close()

# plt.loglog(opt.freq, z_target, '--', color= 'black')
# plt.loglog(opt.freq, np.abs(best_z))
# plt.title('Best Solution',fontsize = 16)
# plt.xlabel('Freq in Hz',fontsize = 16)
# plt.ylabel('Impedance in Ohms', fontsize = 16)
# plt.legend(['Target Z', 'Best Solution'],prop={'size': 12})
# plt.grid(True, which = 'Both')
#plt.savefig("Final_Test_100_Gen_2.png")
#plt.show()

