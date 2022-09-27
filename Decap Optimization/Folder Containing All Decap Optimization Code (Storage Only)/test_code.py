import copy
import numpy as np
from copy import deepcopy
import ShapePDN as pdn1
from config2 import Config
import collections
import matplotlib.pyplot as plt
import math
# def sweep_check(self, min_zero_map):  # should use this at some point
#     improve_bool = True
#     stop_ind = 0
#     while improve_bool is True:
#         current_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # number of caps in the initial min_zero_map
#         print('Before check, decap map is:', min_zero_map)
#         for ind, _ in enumerate(min_zero_map[stop_ind::]):  # iterating through each decap in min map
#             holder = copy.deepcopy(min_zero_map)  # make copy
#             if min_zero_map[ind + stop_ind] != 0:  # if port is not empty
#                 holder[ind + stop_ind] = 0  # make port empty
#                 obj = self.f(holder)
#                 if obj < self.best_function:
#                     # if # of capacitors decrease and target met, overwrite min zero map
#                     min_zero_map = copy.deepcopy(holder)  # update to better map
#                     # improve_bool still true
#                     stop_ind = ind
#                     break
#                 else:
#                     holder = copy.deepcopy(min_zero_map)
#         new_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # used to set improve bool
#         if new_min > current_min:
#             print('After check, number of capacitors decreased. Min decap layout is', min_zero_map)
#             print('Checking Again....')
#             improve_bool = True  # not needed but helps me with clarity
#         else:
#             print('After check, number of capacitors did not decrease.')
#             improve_bool = False  # score did not improve, set improve_bool to false. break out of loop
#     return min_zero_map
#
# def sweep_check2(self, min_zero_map):  # should use this at some point
#     improve_bool = True
#     stop_ind = 0
#     remove_order = np.flip(np.argsort(min_zero_map))
#     while improve_bool is True:
#         current_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # number of caps in the initial min_zero_map
#         print('Before check, decap map is:', min_zero_map)
#         for ind, _ in enumerate(remove_order[stop_ind::]):  # iterating through each decap in min map
#             holder = copy.deepcopy(min_zero_map)  # make copy
#             if min_zero_map[remove_order[ind + stop_ind]] != 0:  # if port is not empty
#                 holder[remove_order[ind + stop_ind]] = 0  # make port empty
#                 obj = self.f(holder)
#                 if obj < self.best_function:
#                     # if # of capacitors decrease and target met, overwrite min zero map
#                     min_zero_map = copy.deepcopy(holder)  # update to better map
#                     # improve_bool still true
#                     stop_ind = ind
#                     break
#                 else:
#                     holder = copy.deepcopy(min_zero_map)
#         new_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # used to set improve bool
#         if new_min > current_min:
#             print('After check, number of capacitors decreased. Min decap layout is', min_zero_map)
#             print('Checking Again....')
#             improve_bool = True  # not needed but helps me with clarity
#         else:
#             print('After check, number of capacitors did not decrease.')
#             improve_bool = False  # score did not improve, set improve_bool to false. break out of loop
#     return min_zero_map
#
# def reduce_via_short(mergedL, ports_shorted, ic_port=0):
#
#     # merged L is the L matrix with IC vias merged. IC via inductance assumed as port 0
#     # ports_shorted is a list of already shorted ports (ports with decaps already placed)
#
#     # Currently does not work if there are 2 ports left and you are deciding which via to remove next.
#
#
#     B = np.linalg.inv(mergedL)  # get B matrix
#     Leq_mat = np.zeros(len(ports_shorted)) # holder for storing equivalent inductances
#     short_prio = np.array(ports_shorted)
#     for i in range(len(ports_shorted)):
#         ports_to_short = [j for j in ports_shorted if j != ports_shorted[i]] # short every port except the i'th port
#         B_new = B[np.ix_(ports_to_short, ports_to_short)]  # extract out only the rows and columns to short
#         # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
#         Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
#         L = np.linalg.inv(Beq)
#         Leq = L[0, 0] + L[1, 1] - L[0, 1] - L[1, 0]
#         Leq_mat[i] = Leq
#     Leq_sorted = np.argsort(Leq_mat) # sort inductances from lowest to highest
#     short_prio = (short_prio[np.s_[Leq_sorted]]) # relate L to the port number, still lowest to highest
#     return short_prio
#
#
#
# def short_vias(L_mat, vias_shorted = None, prev_Leq_mat= None):
#
#     #L_mat is the inductance matrix, with ic inductance assumed to be port 1 (index 0) of L_mat
#     # vias_shorted should be an array containing the ports already shorted.
#     # prev_equiv_mat is the equivalent 2x2 inductance matrix for the last port that was shorted
#         # this is for use when you short the next port. THis is to prevent the array from being massive
#         # need to check if this works though, it should.
#
#     short_prio = []
#     short_array = []
#
#     if vias_shorted is None: # if no ports shorted yet.
#         short_array = np.zeros((np.shape(L_mat)[0] - 1))
#         for i in range(np.shape(short_prio)[0] - 1):
#             short_array[i] = L_mat[0,0] + L_mat[i+1,i+1] - 2*L_mat[0,i+1]
#         short_prio = np.argsort(short_array)  # sorts based on lowest Leq to highest
#         short_prio = np.flip(short_prio)      # flip to get highest to lowest
#         short_prio = short_prio + 1
#     else:
#         if np.shape(vias_shorted)[0] == 1: # if only 1 port has been chosen so far.
#             short_array = np.zeros((np.shape(L_mat)[0] - np.shape(vias_shorted)[0]-1))
#             v_s = vias_shorted[0] # port that was shorted
#             vias = np.arange(0,np.shape(L_mat)[0])  # List of all vias (including IC)
#             vias = np.delete(vias, [0,v_s],axis=0)  # delete out IC via and the 1 shorted via
#             for i in range(np.shape(vias)[0]):
#                 s = vias[i] # next port to short
#                 if s == v_s or s == 0:
#                     pass
#                 else:
#                     temp_Lmat = np.array([[L_mat[0,0], L_mat[0,v_s], L_mat[0,s]], [L_mat[v_s,0],
#                                                                         L_mat[v_s,v_s], L_mat[v_s,s]],
#                                            [L_mat[s,0], L_mat[s,v_s], L_mat[s,s]]])
#                     B = np.linalg.inv(temp_Lmat)
#                     B_reduced = np.array([[B[0,0], B[0,1] + B[0,2]], [B[1,0]+ B[2,0], B[1,1] + B[2,2] + B[1,2] + B[2,1]]])
#                     Leq_mat = np.linalg.inv(B_reduced)
#                     short_array[i] = Leq_mat[0,0] + Leq_mat[1,1] - Leq_mat[1,0] - Leq_mat[0,1]
#             short_sorted = np.argsort(short_array)
#             short_prio = (vias[np.s_[short_sorted]])[::-1]
#     return short_prio
#
# def mass_short_vias(mergedL, ports_to_short, ic_port = 0):
#
#     B = np.linalg.inv(mergedL) # get B matrix
#     ports_to_short = [ic_port] + ports_to_short
#
#
#     #This function assumes that ic_port is always port 0.
#     B = B[np.ix_(ports_to_short,ports_to_short)] # extract out only the rows and columns to short
#     # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
#     Beq = np.add.reduceat(np.add.reduceat(B, [0,1], axis=0), [0,1], axis=1)
#     L = np.linalg.inv(Beq)
#     Leq = L[0,0] + L[1,1] - L[0,1] - L[1,0]
#     return Leq
#
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
# def OptionsInit():
#     # Get settings
#     opt = Config()
#     return opt
#
#
# def decap_objects(opt):
#     cap_objs = [pdn1.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors
#     cap_objs_z = copy.deepcopy(cap_objs)
#     cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
#     return cap_objs, cap_objs_z
#
#
# def get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log'):
#     f_transit = fstop * R / Zmax
#     if interp == 'log':
#         freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
#     elif interp == 'linear':
#         freq = np.linspace(fstart, fstop, nf)
#     ztarget_freq = np.array([fstart, f_transit, fstop])
#     ztarget_z = np.array([R, R, Zmax])
#     ztarget = np.interp(freq, ztarget_freq, ztarget_z)
#
#     return freq, ztarget
#
# def find_nearest(a, a0):
#     "Element in nd array `a` closest to the scalar value `a0`"
#     idx = np.abs(a - a0).argmin()
#     return idx
#
#
# opt = OptionsInit()  # Create settings reference
# cap_objs, cap_objs_z = decap_objects(opt)  # generate capacitor objects and there z parameters
# initial_z = opt.input_net.z
#
# R = .005
# Zmax = .024
# freq3, z_target3 = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log')
#
# break_f = [1e6,10e6,20e6]
# holder = [0,0,0]
# for i in range(len(holder)):
#     holder[i] = find_nearest(freq3, break_f[i])
#
# holder = [0] + holder
#
# decades = len(holder) - 1
# area_above = []
# for i in range(decades):
#     pts_above = np.nonzero(np.greater(np.abs(initial_z[holder[i]:holder[i+1]+1,0,0]),z_target3[holder[i]:holder[i+1]+1]))
#     pts_above = pts_above[0] + holder[i]
#     area_above = area_above + [np.sum(np.abs(initial_z[np.s_[pts_above],0,0]) - z_target3[np.s_[pts_above]])]
#
# cap_list = [10,9,8,7,6,5,4,3,2,1]
# weights = [i/sum(area_above) for i in area_above]
# print(weights)
# print([i * 100 for i in weights])
# prob = [weights[0]/2,weights[0]/2, weights[1]/5, weights[1]/5,weights[1]/5,weights[1]/5,weights[1]/5,weights[2]/3,weights[2]/3,weights[2]/3]
#
# # pop = np.array([np.zeros(50+ 1)] * 50)
# # # stores each solutio
# # solo = np.zeros(50 + 1)
# #
# # # Each var is a solution
# # var = np.zeros(50)
# #
# # # randomly generate population with weights
# # for p in range(0, 50):
# #
# #     for i in range(50):
# #         var[i] = np.random.choice(cap_list,p = prob)
# #     for i in range(np.shape(var)[0]):
# #         solo[i] = var[i].copy()
# #
# #     pop[p] = solo.copy()
# #
# # np.save('Calculated Weighted Pop 5 Different Target.npy', pop)
# # test = np.load('Calculated Weighted Pop 5 Different Target.npy')
# # print(np.count_nonzero(pop == test))
# # for i in test:
# #     print(collections.Counter(i.tolist()))
#
# #
# #
# for i in range(len(area_above)):
#    print(area_above[i]/sum(area_above) * 100)
#
# plt.loglog(freq3,z_target3, 'r')
# plt.loglog(freq3, np.abs(initial_z[:,0,0]),'black')
# plt.axvline(x=1000000)
# plt.axvline(x=10000000)
# #plt.axvline(x=20000000)
#
# plt.grid(which = 'both')
# plt.xlabel('Frequency in Hertz', fontsize = 24)
# plt.ylabel('Impedance in Ohms', fontsize = 24)
# plt.title('Weight Distribution Test',fontsize = 24)
# plt.legend(['Impedance Target', 'Impedance with No Decaps'],fontsize = 24)
# plt.show()
# pop = np.array([np.zeros(50+ 1)] * 50)
#
# # stores each solutio
# # solo = np.zeros(50 + 1)
# #
# # # Each var is a solution
# # var = np.zeros(50)
# # # randomly generate population
# # for p in range(0, 50):
# #
# #     for i in range(50):
# #         var[i] = np.random.randint(1,11)
# #
# #     for i in range(np.shape(var)[0]):
# #         solo[i] = var[i].copy()
# #
# #     pop[p] = solo.copy()
# #
# # np.save('Ran Pop Size 50 5.npy', pop)
# # test = np.load('Ran Pop Size 50 5.npy')
# # print(np.count_nonzero(pop == test))
# # for i in test:
# #     print(i)
# #
#
# # cap_list = [1,2,3,4,5,6,7,8,9,10]
# # prob = [.2,.2,.2, .06,.06,.06, .06,.06,.05,.05]
# # pop = np.array([np.zeros(50+ 1)] * 50)
# #
# # # stores each solutio
# # solo = np.zeros(50 + 1)
# #
# # # Each var is a solution
# # var = np.zeros(50)
# #
# # # randomly generate population with weights
# # for p in range(0, 50):
# #
# #     for i in range(50):
# #         var[i] = np.random.choice(cap_list,p = prob)
# #     for i in range(np.shape(var)[0]):
# #         solo[i] = var[i].copy()
# #
# #     pop[p] = solo.copy()
# #
# # np.save('Weighted Random Pop 5.npy', pop)
# test = np.load('Ran Pop Size 50 1.npy')
# holder = [4, 2, 1, 5, 3, 6, 5, 3, 8, 10, 4, 6, 5, 6, 2, 1, 7, 5, 2, 10, 3, 10, 7, 2, 1, 3, 10, 3, 2, 7, 7, 8, 6, 10, 8, 3, 9, 9, 8, 8, 1, 2, 1, 7, 6, 6, 5, 8, 8, 8]
# holder = np.array(holder)
# for i in test:
#     if np.equal(holder,i[0:50]).all():
#         print('here')
#         print(i)
#     #print(collections.Counter(i.tolist()))

#test = [9, 8, 8, 8, 3, 2, 1, 9, 9, 3, 6, 1, 2, 9, 9, 8, 6, 4, 2, 4, 9, 7, 3, 9, 1, 2, 5, 4, 2, 10, 3, 5, 4, 1, 1, 3, 4, 8, 9, 4, 10, 8, 1, 10, 6, 7, 3, 10, 1, 10]
#print(test)

#Set target

def get_target_z_RL(R, Zmax, fstart=1e4, fstop=20e6, nf=201, interp='log'):
    f_transit = fstop * R / Zmax
    if interp == 'log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return freq, ztarget

# # 50 Cap Test 1
R = 0.01
Zmax = 0.024

## 50 Cap Test 2
# R = .012
# Zmax = 0.0525

## 50 Cap Test 3
# R = 0.009
# Zmax = 0.022

freq, z_target = get_target_z_RL(R, Zmax, fstart=1e4, fstop=20e6, nf=201, interp='log')

for i in range(len(freq)):
    print(freq[i])
    print(z_target[i])