import numpy as np
import matplotlib.pyplot as plt

import pdn
from config2 import Config
import copy
import skrf as rf
from pop_preprocess import *
from pdn_class import PDN
import os
def decap_objects(opt):
    # Store capacitor library as their impedances
    cap_objs = [pdn.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors, 1 to 10, high end is not inclusive [x,y) so need + 1
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z

def OptionsInit():
    # Get settings
    opt = Config()
    return opt

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


# z = np.load('new_data_test_to_compare_methods/75 Caps 5.npz')['z']
# brd = PDN()
# brd.save2s(z, '75 Caps 5')

Freq = rf.frequency.Frequency(start=.01e6 / 1e6, stop=20e6/ 1e6, npoints=201,
                                   unit='mhz', sweep_type='log')  # Used to set interpolation
z= rf.Network('new_data_test_to_compare_methods/25 Caps 3.s26p').interpolate(Freq).z

##### Set settings ######
opt = OptionsInit()
_,cap_objs_z = decap_objects(opt)


#### Do pre-processing steps (Combine these all into pop_preprocess file) #####
L_mat = get_L_mat_from_Z_mat(z)
short_prio = loop_short_all_ports(L_mat)[0]
print(short_prio)
print(L_mat[short_prio[:],short_prio[:]])
single_sol = np.zeros(np.shape(short_prio)[0], dtype= int)

z_max = 0
z_max_sol = []
z_min = 1000000000
z_min_sol = []
z_all = 0
z_plot = []
for i in range(np.shape(short_prio)[0]):

    test_sol = np.copy(single_sol)
    test_sol[short_prio[0:i+1]] = 1
    z_test = pdn.new_connect_n_decap(z, test_sol, cap_objs_z, opt)
    print('SOL',test_sol)
    print('HERE', abs(z_test)[-1])

    if np.abs(z_test)[-1] > z_max:
        z_max = np.abs(z_test)[-1]
        z_max_sol = np.copy(test_sol)
        #print('z_max',z_max)
    if np.abs(z_test)[-1] < z_min:
        z_min = np.abs(z_test)[-1]
        z_min_sol = np.copy(test_sol)
        #print('z_min', z_min)
    if np.count_nonzero(test_sol) == np.shape(short_prio)[0]:
        z_full = np.abs(z_test)[-1]
    z_plot.append(np.abs(z_test)[-1])

print('z_min',z_min)
print(z_min_sol)
print('min num',np.count_nonzero(z_min_sol))
print('z_max', z_max)
print(z_max_sol)
print('max num',np.count_nonzero(z_max_sol))
print('z full', z_full)

z_target = get_target_z_RL(0.033,0.033,opt)

test = np.zeros_like(single_sol,dtype=int)
test2 = np.zeros_like(single_sol,dtype=int)
test3 = np.zeros_like(single_sol,dtype=int)
test4 = np.zeros_like(single_sol,dtype=int)

test[short_prio[0]] = 1
test2[short_prio[0:2]] = 1
test3[short_prio[0:12]] = 1
test4[short_prio[0::]] = 1

# plt.loglog(opt.freq, np.abs(z_target), linewidth = 4)
# plt.loglog(opt.freq, np.abs(z[:,0,0]), linewidth = 4, color = 'r')
# plt.xlabel('Frequency in Hz', fontsize = 28)
# plt.ylabel('Impedance in Ohms', fontsize = 28)
# plt.title('Board Impedance with No Decoupling', fontsize = 28)
# plt.xticks(fontsize = 22)
# plt.yticks(fontsize = 22)
# plt.grid(which = 'both')
# plt.legend(['Target Impedance', 'Input Impedance'], fontsize = 24)
# plt.show()

z_see = pdn.new_connect_n_decap(z, test, cap_objs_z, opt)
z_see2 = pdn.new_connect_n_decap(z, test2, cap_objs_z, opt)
z_see3 = pdn.new_connect_n_decap(z, test3, cap_objs_z, opt)
z_see4 = pdn.new_connect_n_decap(z, test4, cap_objs_z, opt)

plt.loglog(opt.freq, np.abs(z_target), linewidth = 4)
plt.loglog(opt.freq, np.abs(z_see), linewidth = 4)
plt.loglog(opt.freq, np.abs(z_see2), linewidth = 4)
plt.loglog(opt.freq, np.abs(z_see3), linewidth = 4)
plt.loglog(opt.freq, np.abs(z_see4), linewidth = 4)
#plt.legend(['Target Impedance', '1 Port with #1 Cap', '2 Ports with #1 Cap', '12 Ports with #1 Cap', 'All 25 Ports with #1 Cap'], fontsize = 24)
plt.title('25 Port Case Example, Effect of SRF on Local Impedance', fontsize = 28)
plt.xlabel('Frequency in Hz', fontsize = 28)
plt.ylabel('Impedance in Ohms', fontsize = 28)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.grid(which = 'both')
plt.show()

# plt.scatter(range(1,26), z_plot, linewidths= 4, c = 'blue')
# plt.xlabel('Number of Ports Filled', fontsize = 28)
# plt.ylabel('Impedance of Final Point in Ohms', fontsize = 28)
# plt.title('Impedance at Final Point, Incrementally Filling with Capacitor 1', fontsize = 28)
# plt.xticks(fontsize = 22)
# plt.yticks(fontsize = 22)
# plt.plot(range(1,26), [0.033] * 25, linewidth = 4, c = 'red', ls = '--')
# #plt.axhline(y = 0.033, c = 'r',ls ='--', line)
# plt.grid(which = 'both')
# plt.legend(['Impedance', 'R = 0.033 mOhms Target'], fontsize = 24)
# plt.show()