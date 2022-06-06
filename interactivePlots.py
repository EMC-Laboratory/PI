import numpy as np
import matplotlib.pyplot as plt
import ast
import copy
import ShapePDN as pdn1
from config2 import Config

def OptionsInit():
    # Get settings
    opt = Config()
    return opt


def decap_objects(opt):
    cap_objs = [pdn1.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z

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

# Initialize settings
opt = OptionsInit()  # Create settings reference
cap_objs, cap_objs_z = decap_objects(opt)  # generate capacitor objects and there z parameters
initial_z = opt.input_net.z

# Get file names
BASE_PATH = 'Testing Methods Comparing with Paulis/'

#n1 =  BASE_PATH + 'OG + Paulis Scoring Fixed Ran Pop 1 Board 1 Sols 1.npy.txt'
#n1 =  BASE_PATH + 'Changed Scoring + Paulis Scoring Fixed Ran Pop 1 Board 1 Sols 1.npy.txt'
n1 =  BASE_PATH + 'Shuffle Changed Scoring + Paulis Scoring Fixed Ran Pop 1 Board 1 Sols 1.npy.txt'

# get z targets

# 50 Cap Test 1
R = 0.01
Zmax = 0.024

## 50 Cap Test 2
# R = .012
# Zmax = 0.0525

# 50 Cap Test 3
# R = 0.009
# Zmax = 0.022

freq, z_target = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log')


# Read solutions from files and return unique ordered list of solutions
f = open(n1)
sols = f.readlines()
data = [ast.literal_eval(sol) for sol in sols]
unique_ordered_sols = []
unique_index = []
for ind, i in enumerate(data):
    if i not in unique_ordered_sols:
        unique_ordered_sols = unique_ordered_sols + [copy.copy(i)]
        unique_index = unique_index + [ind]

# Get list of impedances
z_list = [None] * len(unique_ordered_sols)

for i in range(len(z_list)):
    copy_sol = copy.copy(unique_ordered_sols[i])
    z = pdn1.new_connect_n_decap(initial_z, copy_sol, cap_objs_z, opt)
    z_list[i] = copy.copy(z)

# plot
bool = 1

# while bool == 1:
#     for i in range(len(unique_index)):
#         fig = plt.figure(figsize=(10,8))
#         plt.loglog(freq, np.abs(z_list[i]))
#         plt.loglog(freq, z_target, '--', color = 'r')
#         plt.grid(which= 'both')
#         plt.xlabel('Frequency in Hz', FontSize=16)
#         plt.ylabel('Magnitude Impedance in Ohms', FontSize=16)
#         plt.title('Best Solution in Generation {}'.format(unique_index[i]), FontSize=16)
#         plt.legend(['Solution', 'Impedance Target'], prop={"size": 18}, loc='upper left')
#         ax = plt.gca()
#         ax.tick_params(axis="x", labelsize=20)
#         ax.tick_params(axis="y", labelsize=20)
#         plt.show()
#     bool = int(input('Enter 1 to Repeat, 0 to end: '))

gen_found = 0
while bool == 1:
    prev_sol = None
    prev_map = None
    for i in range(len(unique_index)):

        fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12, 8))
        ax1.loglog(freq, np.abs(z_list[i]), color = 'black')
        ax1.loglog(freq, z_target, '--', color = 'r')
        if prev_sol is not None:
            ax1.loglog(freq, np.abs(prev_sol), ':', color ='green')
        ax1.grid(which= 'both')
        ax1.set_xlabel('Frequency in Hz', FontSize=12)
        ax1.set_ylabel('Magnitude Impedance in Ohms', FontSize=12)
        ax1.set_title('Best Solution in Generation {}'.format(unique_index[i]), FontSize=14)
        if np.count_nonzero(np.greater(np.abs(z_list[i]),z_target)) == 0:
            ax1.legend(['Solution (Target Met. Num Caps: {})'.format(np.count_nonzero(unique_ordered_sols[i])), 'Impedance Target',
                        'Prev Z Met Solution (Gen {}, Num Caps: {})'.format(gen_found,np.count_nonzero(prev_map))], prop={"size": 13},
                       loc='upper left')
        else:
            ax1.legend(['Solution', 'Impedance Target',
                        'Prev Z Met Solution (Gen {}, Num Caps: {})'.format(gen_found, np.count_nonzero(prev_map))],
                       prop={"size": 13},
                       loc='upper left')
        ax1.tick_params(axis="x", labelsize=20)
        ax1.tick_params(axis="y", labelsize=20)

        ax2.loglog(freq, np.abs(z_list[i]), color='black')
        ax2.loglog(freq, z_target, '--', color='r')

        if prev_sol is not None:
            ax2.loglog(freq, np.abs(prev_sol), ':', color='green')
        ax2.grid(which='both')
        ax2.set_xlabel('Frequency in Hz', FontSize=12)
        ax2.set_ylabel('Magnitude Impedance in Ohms', FontSize=12)
        ax2.set_title('Best Solution in Generation {}'.format(unique_index[i]), FontSize=14)
        if np.count_nonzero(np.greater(np.abs(z_list[i]), z_target)) == 0:
            ax2.legend(['Solution (Target Met. Num Caps: {})'.format(np.count_nonzero(unique_ordered_sols[i])),
                        'Impedance Target',
                        'Prev ZMet Solution (Gen {}, Num Caps: {})'.format(gen_found,
                                                                                 np.count_nonzero(prev_map))],
                       prop={"size": 13},
                       loc='lower right')
        else:
            ax2.legend(['Solution', 'Impedance Target',
                        'Prev Z Met Solution (Gen {}, Num Caps: {})'.format(gen_found,
                                                                                 np.count_nonzero(prev_map))],
                       prop={"size": 13},
                       loc='lower right')

        ax2.tick_params(axis="x", labelsize=20)
        ax2.tick_params(axis="y", labelsize=20)
        ax2.set_xlim([8e6, 20e6])
        ax2.set_ylim([.7e-2, .024])
        plt.show()
        if np.count_nonzero(np.greater(np.abs(z_list[i]),z_target)) == 0:
            prev_sol = np.copy(np.abs(z_list[i]))
            prev_map = copy.copy(unique_ordered_sols[i])
            gen_found = unique_index[i]


    bool = int(input('Enter 1 to Repeat, 0 to end: '))
#
# while bool == 1:
#     prev_sol = None
#     prev_map = None
#     for i in range(len(unique_index)):
#
#         fig = plt.figure(figsize=(10, 8))
#         plt.loglog(freq, np.abs(z_list[i]), color='black')
#         plt.loglog(freq, z_target, '--', color='r')
#         if prev_sol is not None:
#             plt.loglog(freq, np.abs(prev_sol), ':', color='green')
#         plt.grid(which='both')
#         plt.xlabel('Frequency in Hz', FontSize=16)
#         plt.ylabel('Magnitude Impedance in Ohms', FontSize=16)
#         plt.title('Best Solution in Generation {}'.format(unique_index[i]), FontSize=18)
#         if np.count_nonzero(np.greater(np.abs(z_list[i]), z_target)) == 0:
#             plt.legend(['Solution (Target Met. Num Caps: {})'.format(np.count_nonzero(unique_ordered_sols[i])),
#                         'Impedance Target',
#                         'Prev Target Met Solution (Gen {}, Num Caps: {})'.format(gen_found,
#                                                                                  np.count_nonzero(prev_map))],
#                        prop={"size": 16},
#                        loc='upper left')
#         else:
#             plt.legend(['Solution', 'Impedance Target',
#                         'Prev Target Met Solution (Gen {}, Num Caps: {})'.format(gen_found,
#                                                                                  np.count_nonzero(prev_map))],
#                        prop={"size": 16},
#                        loc='upper left')
#         ax = plt.gca()
#         ax.tick_params(axis="x", labelsize=20)
#         ax.tick_params(axis="y", labelsize=20)
#         plt.show()
#         if np.count_nonzero(np.greater(np.abs(z_list[i]), z_target)) == 0:
#             prev_sol = np.copy(np.abs(z_list[i]))
#             prev_map = copy.copy(unique_ordered_sols[i])
#             gen_found = unique_index[i]
#
#     bool = int(input('Enter 1 to Repeat, 0 to end: '))

f.close()

