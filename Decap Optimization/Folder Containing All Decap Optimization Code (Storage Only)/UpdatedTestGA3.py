# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:47:11 2020

@author: lingzhang0319
"""

#from ModGA_FindInitial_v3 import geneticalgorithm as ga
from ModGA_FindInitial_v5 import geneticalgorithm as ga
import ShapePDN as pdn
import matplotlib.pyplot as plt
from config2 import Config
import os
from pop_preprocess import *
import copy

def f(X):

    # for future, maybe changing the generated numpy array to always be int so that I don't have to cast the type,
    # i can remore the astype line and change deepcopy --> copy

    z = z2.copy()
    decap_map = X.copy()
    decap_map = decap_map.astype(int)

    map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
    z_solution = np.abs(map_z)

    ind_fail = 0

    if np.count_nonzero(np.greater(z_solution, z_target)) == 0: # <---- If target met

        reward = len(decap_map) - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports
        #reward = math.log(reward,10)
        #last_z_mod = 1 + z_solution[-1]/z_target[-1]               # <----- How close solution is to tarrget
        #reward = reward * last_z_mod

    else:
        reward = -(np.max((z_solution - z_target) / z_target))
        ind_fail = np.where(z_solution > z_target)[0][0]

    return -reward, ind_fail

# def f(X):
#
#     # for future, maybe changing the generated numpy array to always be int so that I don't have to cast the type,
#     # i can remore the astype line and change deepcopy --> copy
#
#     z = copy.copy(z2)
#     decap_map = copy.copy(X)
#     decap_map = decap_map.astype(int)
#
#     map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
#     z_solution = np.abs(map_z)
#
#     if np.count_nonzero(np.greater(z_solution, z_target)) == 0: # <---- If target met
#
#         reward = len(decap_map) - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports
#         #reward = math.log(reward,10)
#         #last_z_mod = 1 + z_solution[-1]/z_target[-1]               # <----- How close solution is to tarrget
#         #reward = reward * last_z_mod
#
#     else:
#         reward = -(np.max((z_solution - z_target) / z_target))
#
#     return -reward



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


#### THIS IS SOMETHING THAT HAS TO BE INVESTIGATED ####
# If i make a deepcopy z2, and in my f(X) to calculate score do a deepcopy of z2, it is faster than doing a deepcopy of z

z = copy.deepcopy(opt.input_net.z)
z2 = copy.deepcopy(opt.input_net.z)


##### Do pre-processing steps (Combine these all into pop_preprocess file) #####
L_mat = get_L_mat_from_Z_mat(z)
short_prio = loop_short_all_ports(L_mat)[0]
input_data = np.load('Precise Targets Unsorted.npz')
targets = input_data['z_targets']
z_target = np.copy(targets[61])
#for i in range(len(z_target)):
#     z_target[i] = 0.033
p_vector, srf_dir = get_probability_vector(z, cap_objs_z, opt, short_prio)
######
R= 0.005
Zmax = 0.022
freq, z_target = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=opt.nf, interp='log')
print(z_target)

target_type = identify_R_or_RL_Target(z_target)
if target_type == 'R':
    #caps_group1, num_group1 = R_type_target(z_target, z, cap_objs_z, opt, short_prio, check_all=False)
    #caps_group1, num_group1 = R_type_target2(z_target, z, cap_objs_z, opt, short_prio)
    caps_group1, num_group1 = R_type_target3(z_target, z, cap_objs_z, opt, short_prio, srf_dir)

else:
    caps_group1, num_group1 = RL_type_target(z_target, z, cap_objs_z, opt, short_prio,check_all=False)
    caps_group1, num_group1 = RL_type_target2(z_target, z, cap_objs_z, opt, short_prio,check_all=False)
    #caps_group1 = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]],dtype = int)
    #num_group1 = np.array([[19,2,35,36,51,48,46,48,68,69],[2,2,14,36,37,58,53,62,75,75]],dtype = int)
print('Caps Group 1:', caps_group1)
print('Num Group 1:', num_group1)

# Set GA up
num_ports = z.shape[1] - 1
varbound = np.array([[0, 10]] * num_ports)
algorithm_param = {'max_num_iteration': 50,
                   'population_size': 50,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3 , #from .3
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': None}

seed_sol = None
record_solutions = False
f_conv = None
f_sols = None
path = 'Data Collection Test/'
#### Set record solutions to true to record all solutions and the convergence curves
if record_solutions:
    f_conv_name = "Test 75 Caps RL Target All New Functions Conv 3.txt"
    f_conv_name = os.path.join(path,f_conv_name)
    if os.path.isfile(f_conv_name):
        raise FileExistsError("The file for writing convergence curve scores, {}, already exists".format(f_conv_name))
    else:
        f_conv = open(f_conv_name, "w+")

    f_sols_name = "Test 75 Caps RL Target All New Functions Sols 3.txt"
    f_sols_name = os.path.join(path,f_sols_name)
    if os.path.isfile(f_sols_name):
        raise FileExistsError("The file for writing decap solutions, {}, already exists".format(f_sols_name))
    else:
        f_sols = open(f_sols_name, "w+")

model = ga(function=f,
           dimension=num_ports,
           caps_group1 = caps_group1,
           num_group1 = num_group1,
           variable_type='int',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param,
           seed_sol= None,
           L_mat = None,
           record_solutions = record_solutions,
           f_sols = f_sols,
           short_prio = short_prio,
           p_vector= p_vector,
           target_type = target_type)

t1 = time.time()
model.run()
if record_solutions:
    f_conv.write("Time taken =" + str(time.time() - t1) + "\n")

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

# Get final impedance targets
final_z = pdn.new_connect_n_decap(z, decap_solution, cap_objs_z, opt)
best_z = pdn.new_connect_n_decap(z, sol_meeting_target, cap_objs_z, opt)

# These are the eact same solutions so its kinda pointless
print('Best Solution in Final GA Population:', decap_solution.tolist())
print('Best Overall Solution:', sol_meeting_target.tolist())
print('Time Taken =', time.time() - t1 )
# Write things of interest to file for data collection

if record_solutions:
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
    f_sols.close()

# plt.loglog(freq, z_target, '--', color= 'black')
# plt.loglog(freq, np.abs(final_z))
# plt.loglog(freq, np.abs(best_z))
# plt.title('Final Solutions',fontsize = 16)
# plt.xlabel('Freq in Hz',fontsize = 16)
# plt.ylabel('Impedance in Ohms', fontsize = 16)
# plt.legend(['Target Z', 'Best Solution in Final Population', 'Best Solution Satisfying Target Z'],prop={'size': 12})
# plt.grid(True, which = 'Both')
# #plt.savefig("Final_Test_100_Gen_2.png")
# plt.show()

