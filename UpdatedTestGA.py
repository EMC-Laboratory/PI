# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:47:11 2020

@author: lingzhang0319
"""

import numpy as np
#from Online_GA_Unchanged import geneticalgorithm as ga
#from modOnlineGAedit import geneticalgorithm as ga
#rom modOnlineGA import geneticalgorithm as ga
#from UpdatedModGA import geneticalgorithm as ga
from ModGA_FindInitial import geneticalgorithm as ga
from math import pi
import ShapePDN as pdn
import matplotlib.pyplot as plt
import time
from config2 import Config
import copy
import os
import pop_preprocess

def get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log'):
    f_transit = fstop * R / Zmax
    if interp == 'log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return freq, ztarget

def f(X):

    # for future, maybe changing the generated numpy array to always be int so that I don't have to cast the type,
    # i can remore the astype line and change deepcopy --> copy
    z = copy.deepcopy(z2)
    decap_map = copy.copy(X)
    decap_map = decap_map.astype(int)

    map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
    z_solution = np.abs(map_z)

    #print(np.where(np.greater(z_solution, z_target)))
    #print(np.count_nonzero(np.greater(z_solution,z_target)))
    if np.count_nonzero(np.greater(z_solution, z_target)) == 0: # <---- If target met

        reward = len(decap_map) - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports
        #reward = math.log(reward,10)
        #last_z_mod = 1 + z_solution[-1]/z_target[-1]               # <----- How close solution is to tarrget
        #reward = reward * last_z_mod

    else:
        reward = -(np.max((z_solution - z_target) / z_target))
        # reward = (np.max((z_solution - z_target) / z_target) )          # <--- decrease largest z pt
        # ports_shorted = np.nonzero(decap_map)[0].tolist()
        # ports_shorted.sort()
        # true_ports_shorted = np.flip(short_prio)[0:np.count_nonzero(decap_map)].tolist()
        # true_ports_shorted.sort()
        # multiplier = short_some_ports(L_mat, true_ports_shorted) / short_some_ports(L_mat, ports_shorted)
        # reward = -(reward * multiplier)
        #reward = -1/(np.min(np.where(np.greater(z_solution,z_target))[0]))
        #reward = -(np.max((z_solution - z_target)/z_))
        # reward = math.inf
        # for i in range(len(z_solution)):
        #     temp_reward = (z_solution[i] - z_target[i])/z_target[i]
        #     reward = temp_reward if (temp_reward < reward and temp_reward > 0) else reward
        # reward = -reward
    return -reward

def f2(X):

    # for future, maybe changing the generated numpy array to always be int so that I don't have to cast the type,
    # i can remore the astype line and change deepcopy --> copy
    z = copy.deepcopy(z2)
    decap_map = copy.copy(X)
    decap_map = decap_map.astype(int)

    map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
    z_solution = np.abs(map_z)

    # print(np.count_nonzero(np.greater(z_solution,z_target)))
    if np.count_nonzero(np.greater(z_solution, z_target)) == 0:  # <---- If target met

        reward = len(decap_map) - np.count_nonzero(decap_map) + 1  # <----- Number of empty ports
        # reward = math.log(reward,10)
        # last_z_mod = 1 + z_solution[-1]/z_target[-1]               # <----- How close solution is to tarrget
        # reward = reward * last_z_mod

    else:
        score_holder = 0
        for index in range(len(z_solution)):
            add_score = abs(z_solution[index] - z_target[index]) if abs(z_solution[index]) >= abs(
                z_target[index]) else 0
            score_holder = score_holder + add_score
        pts_above = np.count_nonzero(np.greater(z_solution, opt.ztarget))
        pts_above = pts_above if pts_above > 0 else 1
        reward = -1 * score_holder / pts_above
        # reward = math.inf
        # for i in range(len(z_solution)):
        #     temp_reward = (z_solution[i] - z_target[i])/z_target[i]
        #     reward = temp_reward if (temp_reward < reward and temp_reward > 0) else reward
        # reward = -reward
    return -reward


def get_L_mat_from_Z_mat(Z_mat):
    # Z_mat is assumed fxNxN impedance matrix where N are the ports and N > 1 and f are the frequency points
    # this function constructs an L matrix from Z matrix. L matrix will be of shape NxN including self and mutual L
    Zmat_copy = np.copy(Z_mat)
    imag_zmat = np.imag(Zmat_copy)
    L_mat = np.delete(imag_zmat, range(np.shape(Zmat_copy)[0]-1), axis = 0) / (2 * pi * 20e6)
    L_mat = np.reshape(L_mat, (np.shape(Zmat_copy)[1], np.shape(Zmat_copy)[1]))

    return L_mat

def short_all_vias(L_mat):
    # L_mat is an MxM matrix

    # function to determine order the order in which the M ports of the L matrix should be left open

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0])) # get inverse
    shorted_Leq = np.ndarray((1,np.shape(short_copy)[0]-1))
    for i in range(1, np.shape(short_copy)[0]):

        ports_to_short = [j for j in range(1,np.shape(short_copy)[0]) if
                          j != i]  # short every port except the i'th port to see the effect of leaving 1 unshorted
        ports_to_short = [0] + ports_to_short # get ic port

        B_new = B[np.ix_(ports_to_short, ports_to_short)]  # extract out only the rows and columns to short

        # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
        Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
        Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0])) # should end up as a 2x2 always
        shorted_Leq[0,i-1] = Leq[0,0] + Leq[1,1] - Leq[0,1] - Leq[1,0]

    # sorts the Leq in increasing L. This tells you which port, if you left open, would result in the lowest Leq seen
    # looking into port 1
    short_prio = np.argsort(shorted_Leq)

    return short_prio


def short_all_vias2(L_mat):
    # L_mat is an MxM matrix

    # function to determine order the order in which the M ports of the L matrix should be left open
    # not looped to find the best

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0]))  # get inverse
    shorted_Leq = np.ndarray((1, np.shape(short_copy)[0] - 1))
    for i in range(1, np.shape(short_copy)[0]):
        ports_to_short = [i]
        ports_to_short = [0] + ports_to_short  # get IC observation port into L matrix
        B_new = B[np.ix_(ports_to_short, ports_to_short)]
        # extract out only the rows and columns of ports to short and observation port

        # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
        Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
        Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0]))  # should end up as a 2x2 always
        shorted_Leq[0, i - 1] = Leq[0, 0] + Leq[1, 1] - Leq[0, 1] - Leq[1, 0]

    # sorts the Leq in increasing L. This tells you which port, if you short, would give you what equivalent L.
    short_prio = np.argsort(shorted_Leq)

    return short_prio



def loop_short_all_vias(L_mat):
    # L_mat is an MxM matrix

    # function to determine order the order in which the M ports of the L matrix should be left open
    # not looped to find the best

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0]))  # get inverse
    shorted_Leq = np.ndarray((1, np.shape(short_copy)[0] - 1))
    short_prio = np.empty_like(shorted_Leq, dtype= int)
    ports_not_open = list(range(1,np.shape(short_copy)[0]))

    for x in range(1, np.shape(short_copy)[0]):
        ports_open_tracker = []
        temp_Leq = np.ndarray((1, len(ports_not_open) ))
        pos_tracker = 0

        for i in ports_not_open:

            if len(ports_not_open) != 1:
                ports_to_short = [j for j in ports_not_open if j != i]
            else:
                ports_to_short = [j for j in ports_not_open]
            # short every port except ports already calculated as open previously

            ports_to_short = [0] + ports_to_short
            # get ports that should be shorted + IC ports

            B_new = B[np.ix_(ports_to_short, ports_to_short)]
            # extract out only the rows and columns of ports to short and observation port

            Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
            # merge ports by adding the corresponding rows and columns, assuming IC port is index 0

            Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0]))
            # should end up as a 2x2 always

            temp_Leq[0, pos_tracker] = Leq[0, 0] + Leq[1, 1] - Leq[0, 1] - Leq[1, 0]
            ports_open_tracker = ports_open_tracker + [i]
            pos_tracker = pos_tracker + 1

        sorted_temp_Leq = np.argsort(temp_Leq)[0]
        short_prio[0,x-1] = ports_open_tracker[sorted_temp_Leq[-1]]
        # record the port that if not shorted, gives highest Leq (therefore should be shorted)
        ports_not_open.remove(ports_open_tracker[sorted_temp_Leq[-1]])


    short_prio = short_prio - 1
    short_prio = np.flip(short_prio)
    return short_prio



def loop_short_all_vias2(L_mat):
    # L_mat is an MxM matrix

    # function to determine order the order in which the M ports of the L matrix should be shorted
    # Not looped for every iteration, just one

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0]))  # get inverse
    shorted_Leq = np.ndarray((1, np.shape(short_copy)[0] - 1))
    short_prio = np.empty_like(shorted_Leq, dtype= int)
    ports = list(range(1,np.shape(short_copy)[0]))
    ports_already_shorted = []

    for x in range(1, np.shape(short_copy)[0]):

        ports_shorted_tracker = []
        temp_Leq = np.ndarray((1, len(ports)))
        pos_tracker = 0
        #print('Ports:', ports)
        for i in ports:
            ports_to_short = [0] + [i]
            ports_to_short = ports_to_short + [j for j in ports_already_shorted if j not in ports_to_short]
            ports_to_short.sort()
            # get ports that are already shorted + IC ports
            # the i'th port is checking for the next port to short

            B_new = B[np.ix_(ports_to_short, ports_to_short)]
            # extract out only the rows and columns of ports to short and observation port

            Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
            # merge ports by adding the corresponding rows and columns, assuming IC port is index 0

            Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0]))
            # should end up as a 2x2 always

            temp_Leq[0, pos_tracker] = Leq[0, 0] + Leq[1, 1] - Leq[0, 1] - Leq[1, 0]
            ports_shorted_tracker = ports_shorted_tracker + [i]
            pos_tracker = pos_tracker + 1

        sorted_temp_Leq = np.argsort(temp_Leq)[0]
        short_prio[0,x-1] = ports_shorted_tracker[sorted_temp_Leq[0]]
        # record the port that if not shorted, gives highest Leq (therefore should be shorted)
        ports.remove(ports_shorted_tracker[sorted_temp_Leq[0]])

        #print(ports_shorted_tracker)
        # print(sorted_temp_Leq[0])
        ports_already_shorted = ports_already_shorted + [ports_shorted_tracker[sorted_temp_Leq[0]]]

    short_prio = short_prio - 1  #shift from 1 - N  to 0 - N - 1 where N is the total number of ports
    short_prio = np.flip(short_prio)
    #flipped around to match with other functions. This is the reverse order of ports to be shorted
    return short_prio

def short_some_ports(L_mat, ports_shorted):

    # This funciton gets you the equivalent inductance looking into IC port when shorting the ports given
    # by ports_shorted

    # L_mat is the L matrix
    # ports_shorted is a list of ports to short
    short_copy = np.copy(L_mat)
    ports_shorted_copy = np.copy(ports_shorted)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0]))  # get inverse

    ports_to_short = [0] + ports_shorted_copy
    ports_to_short.sort()

    B_new = B[np.ix_(ports_to_short, ports_to_short)]
    # extract out only the rows and columns of ports to short and observation port

    Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
    # merge ports by adding the corresponding rows and columns, assuming IC port is index 0

    Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0]))
    # should end up as a 2x2 always

    L_seen = Leq[0, 0] + Leq[1, 1] - Leq[0, 1] - Leq[1, 0]

    return L_seen


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
z = copy.deepcopy(opt.input_net.z)
z2 = copy.deepcopy(opt.input_net.z)
L_mat = get_L_mat_from_Z_mat(z)


##### Do pre-processing steps (Combine these all into pop_preprocess file) #####
short_prio = loop_short_all_vias2(L_mat)[0]
input_data = np.load('Precise Targets Unsorted.npz')
targets = input_data['z_targets']
z_target = np.copy(targets[61])
R= 0.01
Zmax = 0.024
freq, _ = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=opt.nf, interp='log')

#print('HERE', z_target)

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

seed_sol = None
record_solutions = False
f_conv = None
f_sols = None
path = 'Data Collection 1-30-22/'
#### Set record solutions to true to record all solutions and the convergence curves
if record_solutions:
    f_conv_name = "Find Initial Sols 5mOhms - 25 mOhms, 33 - 35 Caps Small Caps Best Ports Conv 5.txt"
    f_conv_name = os.path.join(path,f_conv_name)
    if os.path.isfile(f_conv_name):
        raise FileExistsError("The file for writing convergence curve scores, {}, already exists".format(f_conv_name))
    else:
        f_conv = open(f_conv_name, "w+")

    f_sols_name = "Find Initial Sols 5mOhms - 25 mOhms, 33 - 35 Caps Small Caps Best Ports Sols 5.txt"
    f_sols_name = os.path.join(path,f_sols_name)
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
           record_solutions = record_solutions,
           f_sols = f_sols,
           short_prio = short_prio)

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

plt.loglog(freq, z_target, '--', color= 'black')
plt.loglog(freq, np.abs(final_z))
plt.loglog(freq, np.abs(best_z))
plt.title('Final Solutions',fontsize = 16)
plt.xlabel('Freq in Hz',fontsize = 16)
plt.ylabel('Impedance in Ohms', fontsize = 16)
plt.legend(['Target Z', 'Best Solution in Final Population', 'Best Solution Satisfying Target Z'],prop={'size': 12})
plt.grid(True, which = 'Both')
#plt.savefig("Final_Test_100_Gen_2.png")
plt.show()

## Final Check ##
# Check if the min decap and best map sol can have their decap # decreased (Primitively)

#Check if best solution can be improved

#decap_solution = final_check(decap_solution,opt,cap_objs_z)

# Recalculate Impedance, whether it improves or not
#best_z = pdn.new_connect_n_decap(z, decap_solution, cap_objs_z, opt)



############ DON'T WANT TO DELETE YET ##############
# def merge_ports(L_orig, merge_port_list, kept_port=0):
#     # merge_port_list, kept_port index begins with 0
#
#     orig_port_list = list(range(0, L_orig.shape[0])) # get the number of ports
#     del_ports = deepcopy(merge_port_list)            # delete out the ports to be merged
#     del_ports.remove(kept_port)                      # get the port number we want to merge the ports to
#
#     # get the remaining ports (ports that are not to be merged
#     left_port_list = deepcopy(orig_port_list)
#     for a in del_ports:
#         left_port_list.remove(a)
#
#     # get inverse L matrix
#     l_inv = np.linalg.inv(L_orig)
#     if len(orig_port_list) > len(merge_port_list):
#         # merge ports by adding the corresponding rows and columns
#         reduce_list = list([kept_port,merge_port_list[-1]+1]) + list(range(merge_port_list[-1]+2,orig_port_list[-1]+1))
#         l_inv_merge = np.add.reduceat(np.add.reduceat(l_inv, reduce_list,axis=0),reduce_list,axis=1)
#     else:
#         l_inv_merge = np.add.reduceat(np.add.reduceat(l_inv, [0],axis=0),[0],axis=1)
#     l_merge = np.linalg.inv(l_inv_merge)
#     # port_map_orig is the port map to the original port number
#
#     return l_merge
#
#
# BASE_PATH = 'new_data_test/'
# if not os.path.exists(BASE_PATH):
#     os.mkdir(BASE_PATH)
#
# L = np.load(os.path.join('Testing for Moving Vias', '100 Port Boundary Mat Test CCW' +'.npz'))['L']
#
#
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
