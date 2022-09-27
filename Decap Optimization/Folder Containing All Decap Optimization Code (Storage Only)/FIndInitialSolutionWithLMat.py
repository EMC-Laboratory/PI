# Test code for generating initial solution with L mat\

import numpy as np
from math import ceil
import random
from copy import deepcopy
import os
import ShapePDN as pdn
from config2 import Config
import matplotlib.pyplot as plt
def get_target_z_RL(R, Zmax, fstart=1e4, fstop=20e6, nf=201, interp='log'):
    f_transit = fstop * R / Zmax
    if interp == 'log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return ztarget

def reduce_via_short(mergedL, ports_shorted, ic_port=0):

    # merged L is the L matrix with IC vias merged. IC via inductance assumed as port 0
    # ports_shorted is a list of already shorted ports (ports with decaps already placed)

    # Currently does not work if there are 2 ports left and you are deciding which via to remove next.

    B = np.linalg.inv(mergedL)  # get B matrix
    Leq_mat = np.zeros(len(ports_shorted)) # holder for storing equivalent inductances
    short_prio = np.array(ports_shorted)
    for i in range(len(ports_shorted)):
        ports_to_short = [j for j in ports_shorted if j != ports_shorted[i]] # short every port except the i'th port
        B_new = B[np.ix_(ports_to_short, ports_to_short)]  # extract out only the rows and columns to short
        # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
        Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
        L = np.linalg.inv(Beq)
        Leq = L[0, 0] + L[1, 1] - L[0, 1] - L[1, 0]
        Leq_mat[i] = Leq
    Leq_sorted = np.argsort(Leq_mat) # sort inductances from lowest to highest
    short_prio = (short_prio[np.s_[Leq_sorted]]) # relate the sorted L to the port number
    return short_prio

def short_1_via_check(mergedL, ports_shorted, ic_port=0):

    Leq_mat = np.zeros(len(ports_shorted)) # holder for storing equivalent inductances
    short_prio = np.array(ports_shorted)
    for i in range(len(ports_shorted)):
        # 'i' is the current port to short
        port = ports_shorted[i]
        print(port)
        Leq = L[0, 0] + L[port, port] - L[0, port] - L[port, 0]
        Leq_mat[i] = Leq
    Leq_sorted = np.argsort(Leq_mat) # sort inductances from lowest to highest
    short_prio = (short_prio[np.s_[Leq_sorted]]) # relate the sorted L to the port number
    return short_prio


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


def f(X):
    z = deepcopy(opt.input_net.z)
    decap_map = deepcopy(X)
    decap_map = decap_map.astype(int)
    map_z = pdn.new_connect_n_decap(z, decap_map, cap_objs_z, opt)
    z_solution = np.abs(map_z)
    target_met = False
    if np.count_nonzero(np.greater(z_solution, z_target)) == 0: # <---- If target met
        target_met = True
    return target_met, z_solution

#####setup

######### Import settings
opt = OptionsInit()
_,cap_objs_z = decap_objects(opt)

################# Get impedance target
R = .012
Zmax = .026
z_target = get_target_z_RL(R, Zmax, fstart=1e4, fstop=20e6, nf=201, interp='log')

##################### Get L matrix file and do preprocessing ##################
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
##############################################


ports_to_short = list(range(1,101))
remove_order = short_1_via_check(L, ports_to_short)

print(remove_order.tolist())
# Try to find initial solution
via_step = 10
num_solutions = 20
solutions = np.zeros((num_solutions,100), dtype=int)
sol_z = None
sol = None
sol_found = False
for i in [via_step*j for j in range(1,10)]:
    ports_to_fill = remove_order[0:i] - 1
    for j in range(solutions.shape[0]):
        decaps = np.random.random_integers(1, high=10, size=(1,i))
        solutions[j][np.s_[ports_to_fill]] = decaps

    if i != 100:
        for j in solutions:
            print('Solution to evaluate =', j)
            print('Num Caps =', np.count_nonzero(j))
            target_met, z = f(j)
            if target_met:
                print('Solution Found')
                sol = deepcopy(j)
                sol_z = deepcopy(z)
                sol_found = True
                break
            else:
                print('Hold a solution')
                sol = deepcopy(j)
                sol_z = deepcopy(z)

    else:
        print('Did not find a single solution by generating random maps up to a total of 95 decaps')


    if sol_found:
        print('Solution found is:', sol)
        break

f = open('Random Generation for Initial Solution CCW', 'a')

if sol_found:
    f.write('Decap Solution. Num caps =' +  str(np.count_nonzero(sol)) + "\n")
    f.write(str(sol.tolist()))
else:
    f.write('No Decap Solution Found Using N-10 Caps')
f.write("\n,\n")
f.close()

# plot
if sol is not None:
    print(np.shape(sol_z))
    plt.loglog(opt.freq, np.abs(sol_z))
    plt.loglog(opt.freq, z_target)
    plt.xlabel('Frequency in Hz', size = 20)
    plt.ylabel('Impedance in Ohms',size = 20)
    plt.grid(which= 'both')
    plt.title('Impedance vs Frequency',size = 20)
    plt.show()