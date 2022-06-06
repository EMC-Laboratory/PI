from config2 import Config
import ShapePDN as pdn
import copy
import PopInit as Pop
import matplotlib.pyplot as plt
import numpy as np
import random
import skrf as rf
import math as math
import sys


#532968
def OptionsInit():
    # Get settings
    opt = Config()
    return opt


def decap_objects(opt):
    cap_objs = [pdn.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z

def calc_res_peak(rvrm, lvrm, nom_c, nom_esr):
    zo = math.sqrt(lvrm/nom_c)
    q = zo * 1/(nom_esr + rvrm)
    zpeak = zo * q
    return zpeak

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

opt = OptionsInit()  # Create settings reference
cap_objs, cap_objs_z = decap_objects(opt)  # generate capacitor objects and there z parameters
initial_z = opt.input_net.z

R = .005
Zmax = .022
freq, z_target = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log')

# test_file_name = '500 Targets Unshuffled.npz'
# file = np.load(test_file_name)
# targets = file['z_targets']
# sols = file['sols']
# freq = file['Freq']
# target_met_array = np.ndarray((500,1))
#
# input_data = np.load('Precise Targets Unsorted.npz')
# targets = input_data['z_targets']
# decap_map = np.array(
# [0, 1, 1, 4, 1, 0, 0, 0, 1, 0, 6, 0, 8, 2, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 3]
# )
decap_map = [0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 0, 2, 3, 2, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, 2, 6, 0, 0, 0, 8, 3, 3, 0, 2, 2, 7, 2, 2, 0, 0, 4, 0, 0, 2, 2, 0, 2, 3, 0, 0, 1, 2, 0, 0, 0, 0, 2, 1, 5, 0, 0]
sol_z = pdn.new_connect_n_decap(initial_z, decap_map, cap_objs_z, opt)

#plt.loglog(opt.freq, np.abs(initial_z[:,0,0]), linewidth = 3)
plt.loglog(freq, np.abs(sol_z), linewidth = 3)
plt.loglog(freq, z_target, '--', linewidth = 3)
plt.grid(which= 'both')
plt.xlabel('Frequency in Hz', fontsize =24)
plt.ylabel('Impedance in Ohms', fontsize =24)
plt.title('75 Port Case Target Impedance, Known Minimum Solution', fontsize =24)
plt.legend(['Impedance of Solution', 'Target Impedance'], fontsize =24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.show()

test = [0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 0, 2, 3, 2, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, 2, 6, 0, 0, 0, 8, 3, 3, 0, 2, 2, 7, 2, 2, 0, 0, 4, 0, 0, 2, 2, 0, 2, 3, 0, 0, 1, 2, 0, 0, 0, 0, 2, 1, 5, 0, 0]
test2 = np.unique(test, return_counts = True)
print(test2)
print(test2[1]/(np.count_nonzero(test)) * 100)

# plt.loglog(opt.freq, abs(sol_z))
# plt.grid(which= 'both')
# plt.xlabel('Freq', fontsize =24)
# plt.ylabel('Impedance', fontsize =24)
# plt.title('Input Information', fontsize =24)
# plt.legend(['Impedance Target', 'Initial Impedance'], fontsize =24)
# plt.show()
# for i in range(500):
#     print(i)
#     plt.loglog(freq, targets[i])
#     sol_z = pdn.new_connect_n_decap(initial_z, sols[i], cap_objs_z, opt)
#     target_met_array[i] = 1 if np.count_nonzero(np.greater(np.abs(sol_z), targets[i])) == 0 else 0
#
# #['sols', 'sols_found', 'inputs', 'z_targets', 'dist', 'dist_total', 'num_caps'] 
# sols = file['sols']
# sols_found = np.copy(target_met_array)
# inputs = file['inputs']
# z_targets = file['z_targets']
# dist = file['dist']
# dist_total = file['dist_total']
# num_caps = file['num_caps']
# freq = file['Freq']
# np.savez('500 Targets Unshuffled with Freq Vector', sols = sols, sols_found= sols_found, inputs = inputs, z_targets = z_targets, dist = dist, dist_total = dist_total, num_caps = num_caps, freq = freq)

# plt.title('Target Impedances',fontsize = 24)
# plt.xlabel('Frequency in Hz',fontsize = 24)
# plt.ylabel('Impedance in Ohms', fontsize = 24)
# #plt.legend([ 'Prev Impedance Target', 'New Impedance Target', 'Board with No Decoupling Capacitors',' 1 of Capacitor 9', '1 of Capacitor 8',  'Sol 1', 'Run 3', 'Run 4', 'Run 5'],prop={'size': 16}, loc = 2)
# plt.grid(True, which = 'Both')
# ax = plt.gca()
# ax.tick_params(axis="x", labelsize=20)
# ax.tick_params(axis="y", labelsize=20)
# #plt.loglog(freq, np.abs(initial_z[:,0,0]), '-o')
# plt.show()


#
# plt.loglog(freq, z_target)
# #plt.loglog(freq, np.abs(z_ini))
# plt.loglog(freq,np.abs(z1), 'o')
# plt.loglog(freq,np.abs(z2), '-o')
# plt.loglog(freq,np.abs(z3), '--')
# plt.loglog(freq,np.abs(z4), '-.')
# plt.loglog(freq,np.abs(z5), ':')
# plt.title('Fixed Population 1, Best Solution at end of 10th Generation',fontsize = 24)
# plt.xlabel('Frequency in Hz',fontsize = 24)
# plt.ylabel('Impedance in Ohms', fontsize = 24)
# plt.legend([ 'Z Target', 'Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5'],prop={'size': 16}, loc = 4)
# plt.grid(True, which = 'Both')
# ax = plt.gca()
# ax.tick_params(axis="x", labelsize=20)
# ax.tick_params(axis="y", labelsize=20)
# plt.show()
