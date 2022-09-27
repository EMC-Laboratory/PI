
import numpy as np
import ShapePDN as pdn
import matplotlib.pyplot as plt
import time
from config2 import Config
import copy
import os
from math import pi
import random

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
        reward = -((np.max((z_solution - z_target) / z_target) ))          # <--- decrease largest z pt
        #reward = -1/(np.min(np.where(np.greater(z_solution,z_target))[0]))
        #reward = -(np.max((z_solution - z_target)/z_))
        # reward = math.inf
        # for i in range(len(z_solution)):
        #     temp_reward = (z_solution[i] - z_target[i])/z_target[i]
        #     reward = temp_reward if (temp_reward < reward and temp_reward > 0) else reward
        # reward = -reward
    return -reward


def sweep_check(min_zero_map): # should use this at some point
    improve_bool = True
    stop_ind = 0
    best_function = -(50 - np.count_nonzero(min_zero_map) + 1)
    while improve_bool is True:
        current_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # number of caps in the initial min_zero_map
        print('Before check, decap map is:', min_zero_map)
        for ind, _ in enumerate(min_zero_map[stop_ind::]): # iterating through each decap in min map
            holder = copy.deepcopy(min_zero_map) # make copy
            if min_zero_map[ind + stop_ind] != 0: # if port is not empty
                holder[ind + stop_ind] = 0  # make port empty
                obj = f(holder)
                print(obj)
                print('Checking:', holder)
                if obj < best_function:
                    # if # of capacitors decrease and target met, overwrite min zero map
                    min_zero_map = copy.deepcopy(holder) # update to better map
                    # improve_bool still true
                    stop_ind = ind
                    break
                else:
                    holder = copy.deepcopy(min_zero_map)
        new_min = len(min_zero_map) - np.count_nonzero(min_zero_map) # used to set improve bool
        if new_min > current_min:
            print('After check, number of capacitors decreased. Min decap layout is', min_zero_map)
            print('Checking Again....')
            improve_bool = True # not needed but helps me with clarity
        else:
            print('After check, number of capacitors did not decrease.')
            improve_bool = False # score did not improve, set improve_bool to false. break out of loop
    return min_zero_map

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
    # not looped to find the best

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0])) # get inverse
    shorted_Leq = np.ndarray((1,np.shape(short_copy)[0]-1))
    for i in range(1, np.shape(short_copy)[0]):
        ports_to_short = [j for j in range(1,np.shape(short_copy)[0]) if
                          j != i]  # short every port except the i'th port to see the effect of leaving 1 unshorted port
        ports_to_short = [0] + ports_to_short # get IC observation port into L matrix
        B_new = B[np.ix_(ports_to_short, ports_to_short)]
        # extract out only the rows and columns of ports to short and observation port

        # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
        Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
        Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0])) # should end up as a 2x2 always
        shorted_Leq[0,i-1] = Leq[0,0] + Leq[1,1] - Leq[0,1] - Leq[1,0]

    # sorts the Leq in increasing L. This tells you which port, if you left open, would result in the lowest Leq seen
    # looking into port 1
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

        # print('Unopen Ports:', ports_not_open)
        for i in ports_not_open:

            if len(ports_not_open) != 1:
                ports_to_short = [j for j in ports_not_open if j != i]
            else:
                ports_to_short = [j for j in ports_not_open]
            # short every port except ports already calculated as open previously

            ports_to_short = [0] + ports_to_short
            # get ports that should be shorted + IC ports
            # print('Ports to Short:', ports_to_short)

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
    # Not looped for every possible iteration, just 1

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




def reduce_via_short(self, mergedL, ports_shorted, ic_port=0):

    # merged L is the L matrix with IC vias merged. IC via included in array and assumed as port 0
    # ports_shorted is a list of already shorted ports (ports with decaps already placed)

    # Currently does not work if there are 2 ports left and you are deciding which via to remove next.

    B = np.linalg.inv(mergedL)  # get B matrix
    Leq_mat = np.zeros(len(ports_shorted))  # holder for storing equivalent inductances
    short_prio = np.array(ports_shorted)
    for i in range(len(ports_shorted)):
        ports_to_short = [j for j in ports_shorted if
                          j != ports_shorted[i]]  # short every port except the i'th port. Short 1 port at a time
        B_new = B[np.ix_(ports_to_short, ports_to_short)]  # extract out only the rows and columns to short
        # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
        Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
        L = np.linalg.inv(Beq)
        Leq = L[0, 0] + L[1, 1] - L[0, 1] - L[1, 0]
        Leq_mat[i] = Leq
    Leq_sorted = np.argsort(Leq_mat)  # sort inductances from lowest to highest
    short_prio = (short_prio[np.s_[Leq_sorted]])  # relate L to the port number, still lowest to highest
    return short_prio



def shift_mut(short_prio, sol):
    short_copy = np.copy(short_prio)
    new_sol = np.copy(sol)
    mut_rate = .1
    for i in range(np.shape(sol)[0]):
        chance = random.random()
        if chance < mut_rate:
            move = random.randint(1,3)
            loc_ind = np.where(short_copy == i)[0]
            if loc_ind != short_copy[0] or loc_ind != short_copy[-1]:
                direction = random.randint(0,1) # 0 is back, 1 is forward
                if direction == 1:
                    new_loc = loc_ind + move if loc_ind + move < np.shape(short_copy)[0]-1 else np.shape(short_copy)[0]-1
                    if new_sol[new_loc] == 0:
                        new_sol[new_loc] = new_sol[loc_ind]
                        new_sol[loc_ind] = 0
                    else:
                        old_value = new_sol[loc_ind]
                        new_sol[loc_ind] = new_sol[new_loc]
                        new_sol[new_loc] = old_value
                else:
                    new_loc = loc_ind - move if loc_ind - move > 0 else 0
                    if new_sol[new_loc] == 0:
                        new_sol[new_loc] = new_sol[loc_ind]
                        new_sol[loc_ind] = 0
                    else:
                        old_value = new_sol[loc_ind]
                        new_sol[loc_ind] = new_sol[new_loc]
                        new_sol[new_loc] = old_value
    return new_sol

def cust_mut(sol):

    new_sol = np.copy(sol)
    cap_list = np.arange(10+1)
    mut_rate = .1
    for i in range(len(new_sol)):
        chance = np.random.random()
        if chance < mut_rate:
            old_cap = new_sol[i]
            pos = np.where(cap_list == old_cap)[0]

            if pos != 0 and pos != (np.shape(cap_list)[0] - 1):
                new_cap = [cap_list[pos + 1], cap_list[pos - 1]]
                ran = random.random()

                if ran < .5:
                    new_cap = new_cap[0]
                else:
                    new_cap = new_cap[1]

            elif pos == 0:
                if old_cap == 0:
                    new_cap = cap_list[random.randrange(1, np.shape(cap_list)[0])]
                else:
                    new_cap = cap_list[pos + 1]
            else:
                new_cap = cap_list[pos - 1]
            new_sol[i] = new_cap

    return new_sol



###### Set settings ######
opt = OptionsInit()
_,cap_objs_z = decap_objects(opt)
z = opt.input_net.z
z2 = opt.input_net.z

input_data = np.load('Precise Targets Unsorted.npz')
print(list(input_data.keys()))
print(input_data['num_caps'][61])
print(input_data['dist'][61])
print(input_data['sols'][61])

# sols = input_data['sols']
# inputs = input_data['inputs']
# num_caps = input_data['num_caps']
L_mat = get_L_mat_from_Z_mat(z)
# short_prio2 = short_all_vias2(L_mat)[0]
open_prio = loop_short_all_vias(L_mat)[0]
short_prio = loop_short_all_vias2(L_mat)[0]

# for i in range(50):
#     print('Port to Open:', np.flip(short_prio)[i])
#     print('Port to Short:', short_prio2[i], '\n')

# for i in range(50):
#     print(np.flip(short_prio)[i])
#     print(short_prio3[i])


test =  np.array(
    [0, 6, 0, 1, 1, 0, 0, 4, 2, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
)
#test = np.array([7,9,1,1,1,0,1,1,0,1,0,2,0,0,4,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0])

#
#
# sol2 = np.array([0, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 8, 0, 1, 0, 2, 0, 3, 0, 0, 0, 9, 0, 0, 3, 0, 5, 6, 0, 0, 0, 6, 1, 4, 0, 0, 1])
# sol3 = np.array([4,1,1,0,1,1,0,0,0,0,0,1,0,3,1,0,1,1,1,0,0,0,5,1,0,1,1,0,0,0,0,0,0,0,0,1,0,7,0,0,0,0,0,0,0,2,0,0,8,0])
# sol4 = np.array([2,4,0,0,1,7,0,1,3,10,1,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])

#### 17 Caps
#sol = np.array([7,9,1,1,1,0,1,1,0,1,0,2,0,0,4,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0])
#sol = np.array( [7, 0, 1, 1, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 8, 0, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     [6, 1, 1, 1, 1, 3, 1, 3, 0, 0, 3, 3, 2, 5, 0, 4, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 4, 8, 2, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 7]

# for i in range(14):
#     # print('Priority:', i, short_prio[i] )
#     print('Priority:', i)
#     print('Port to Short', short_prio[i] + 1)
#     #print('Port to Open', np.flip(open_prio)[i] + 1,'\n')
#     #print('Decap =', sol[short_prio[i]])
#
#     # print('Port to Not Open', np.flip(open_prio)[i])
#     # print('Decap', sol[np.flip(open_prio)[i]], '\n')
#     #
#     # print('Port to Short', (short_prio2[i]) + 1)
#     # print('Using Ports to Short', sol[short_prio2[i]], '\n')

holder = input_data['z_targets']
num = 37
z_target = holder[num]
print(z_target)
print(input_data['num_caps'][num])
print(input_data['sols'][num])
print(input_data['dist'][num])

#z_target = opt.ztarget
filled_sol = np.ones((50), dtype= int) * 2
single_sol = np.zeros((50), dtype= int)
half_sol = np.zeros((50), dtype= int)
#half_sol[short_prio[0:40]] = 3
single_sol[short_prio[0]] = 2
#half_sol[np.random.choice(np.arange((50), dtype = int), (25), replace = False)] = 3
print(short_prio)
for i in range(50):
    print('Filling', i + 1,'th port')
    test = np.copy(half_sol)
    test[short_prio[0:i+1]] = 10
    print(test)
    z_half =   pdn.new_connect_n_decap(z, test, cap_objs_z, opt)
    if np.abs(z_half)[200] < z_target[200]:
        print('Num Caps Required = ', i + 1)
        half_sol = np.copy(test)
        break
    elif i == 49:
        print('Frequency point 157 not met')
        half_sol = np.copy(filled_sol)
half_sol[short_prio[0:35]] = 1

# for i in range(10):
#     test = np.zeros(50,dtype = int)
#     test[0] = test[0] + i + 1
#     z_single =   pdn.new_connect_n_decap(z, test, cap_objs_z, opt)
#     plt.loglog(opt.freq, np.abs(z_single))
# plt.loglog(opt.freq, z_target)
# plt.grid(which='both')
# plt.xlabel('Frequency in Hz', fontsize=16)
# plt.ylabel('Impedance in Ohms', fontsize=16)
# plt.title('Connecting a Single Capacitor', fontsize=16)
# plt.show()




#
# z_filled =  pdn.new_connect_n_decap(z, filled_sol, cap_objs_z, opt)
# z_single =   pdn.new_connect_n_decap(z, single_sol, cap_objs_z, opt)
# z_half =   pdn.new_connect_n_decap(z, half_sol, cap_objs_z, opt)
#
# plt.loglog(opt.freq, z_target)
# plt.loglog(opt.freq, np.abs(z_filled))
# plt.loglog(opt.freq, np.abs(z_single))
# plt.loglog(opt.freq, np.abs(z_half))
# plt.axvline(x = opt.freq[157])
# #
# plt.xlabel('Frequency', fontsize = 22)
# plt.ylabel('Impedance', fontsize = 22)
# plt.title('Target Z, R = 5 mOhms, Z = 25 mOhms',fontsize = 22)
#
# #plt.title('Effective Max Frequency Range For Decap 1',fontsize = 22)
# plt.legend(['Z Target', 'All Ports Filled with Decap 1', 'Best Port Filled with Decap 1', 'Top 10 Ports Filled with Decap 1'],fontsize = 16)
# plt.grid(which = 'both')
# plt.show()





# print(np.shape(sol))
# for i in range(50):
#     print(short_prio[i], sol[short_prio[i]], sol2[short_prio[i]], sol3[short_prio[i]], sol4[short_prio[i]])
# print(np.count_nonzero(sol), np.count_nonzero(sol2), np.count_nonzero(sol3), np.count_nonzero(sol4))
#test = shift_mut(short_prio, sol)
#test = cust_mut(sol)
#print(test)
# #
# # #

# holder = input_data['z_targets']
# z_target = holder[61]
#
#
sol= np.array(
    [6, 1, 1, 1, 1, 3, 1, 3, 0, 0, 3, 3, 2, 5, 0, 4, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 4, 8, 2, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 7]
)
sol = np.array(
    [1, 3, 2, 3, 4, 2, 8, 9, 0, 2, 0, 3, 0, 0, 0, 4, 2, 4, 4, 2, 2, 0, 0, 7, 2, 3, 4, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 3, 3, 0, 6, 8]
)
sol = np.array(
[2, 2, 2, 4, 4, 0, 3, 0, 0, 5, 2, 0, 0, 2, 0, 0, 1, 4, 2, 0, 9, 2, 7, 4, 10, 4, 0, 0, 3, 0, 7, 0, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0, 6, 0, 0, 2, 5, 0, 1, 0]
)
sol = np.array(
    [2, 3, 2, 3, 4, 2, 8, 9, 0, 2, 0, 3, 0, 0, 0, 4, 2, 4, 4, 2, 2, 0, 0, 7, 2, 3, 4, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0,
     4, 0, 0, 0, 0, 0, 0, 3, 3, 0, 6, 8]
)
sol = np.array(
[0, 2, 3, 8, 0, 7, 0, 4, 3, 0, 0, 0, 0, 9, 0, 7, 4, 4, 4, 0, 3, 0, 3, 2, 2, 1, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 3, 0, 0, 2, 0, 2, 3]
)
#2, 1s
#5, 2s
#9, 3s

z_out =  pdn.new_connect_n_decap(z, sol, cap_objs_z, opt)
print(np.count_nonzero(np.greater(abs(z_out),z_target)) == 0)

plt.loglog(opt.freq, np.abs(z_out))
plt.loglog(opt.freq, z_target)
#plt.axvline(x = opt.freq[153])

#plt.axvline(x = opt.freq[157])
#plt.axvline(x = opt.freq[163])
#plt.axvline(x = opt.freq[190])
plt.grid(which='both')
plt.xlabel('Frequency in Hz', fontsize = 16)
plt.ylabel('Impedance in Ohms', fontsize = 16)
plt.title('Impedane Curve Failing Target', fontsize = 16)

plt.show()


#new_sol = sweep_check(sol)
#print(new_sol)
#if np.count_nonzero(np.equal(new_sol, sol)) != 0:
#     new_sol = sweep_check(new_sol)
#
# print('Final:', new_sol)
# print(sol - new_sol)
# print('Num:', np.count_nonzero(sol))
#
# print('Empmtied:', (np.not_equal(new_sol,sol)))



########## Checking % of group 1
# There is seemingly no pattern
# Possibilities:
#   1.  Either the physics based backing of opening certain ports is wrong. Or opening up ports
#   (rather than picking ports to short) is wrong
#   2.  GA did not find the close enough to the minimum solutions
#   3.  Some target impedances are so easy to satisfy that good port selection doesn't matter
#
#   I think it is a mix of 2 and 3. With that said, targets where 3 doesn't apply, the results don't support
#   1 as being correct (or again just not getting to minimum #)

#
# num_in_ports = np.ndarray((500))
# group_in_ports = np.ndarray((500))
# top_5 = np.flip(short_prio)[0:15]
#
# percentage_group1 = np.ndarray((500))
# for i in range(500):
#     print('R =', inputs[i,0], 'Zmax =', inputs[i,1])
#     print(sols[i][top_5])
#     in_top_5 = np.count_nonzero(sols[i][top_5])
#     num_in_ports[i] = in_top_5
#     group_in_ports[i] = sum([1 for j in sols[i][top_5] if j > 0 and j < 4])
#     percentage_group1[i] = group_in_ports[i] / len(top_5) #if num_in_ports[i] != 0  else 0
# sort_z_max = np.argsort(inputs[:,1])
# plt.scatter(inputs[sort_z_max,1], num_caps[sort_z_max], c = percentage_group1[sort_z_max], cmap = 'magma')
# cbar = plt.colorbar()
# cbar.ax.set_title('Number of Caps in Best 10 Locations')
# plt.show()
#