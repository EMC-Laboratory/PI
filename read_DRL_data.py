# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:57:35 2020

@author: lingzhang0319
"""
from data_generate import *
import numpy as np
import numpy.random as random
from scipy.special import binom
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from copy import deepcopy
from pdn_class import PDN
import time
import os
import pandas as pd
from config2 import Config
def get_target_z_RL(R, Zmax, fstart=1e4, fstop=20e6, nf=132, interp='log'):
    f_transit = fstop * R / Zmax
    if interp =='log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return ztarget

def OptionsInit():
    # Get settings
    opt = Config()
    return opt
opt = OptionsInit()

brd = PDN()

BASE_PATH = 'new_data_test/'
#BASE_PATH = 'new_data_test_to_compare_methods/'



if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

n = '0'

# Plotting from a particular solution
#prio = [86, 51, 60, 22, 6, 36, 79, 53, 35, 40, 90, 37, 14, 42, 17, 12, 41, 92, 30, 8, 16, 23, 13, 93, 33, 68, 85, 18, 1, 97, 77, 76, 3, 44, 70, 43, 49, 5, 21, 84, 47, 73, 67, 2, 80, 64, 81, 95, 94, 63, 48, 11, 69, 62, 75, 66, 78, 7, 58, 38, 57, 72, 26, 39, 91, 88, 10, 99, 59, 71, 29, 4, 31, 89, 65, 27, 28, 15, 24, 46, 56, 32, 52, 20, 34, 100, 54, 45, 74, 98, 61, 96, 19, 55, 25, 87, 83, 50, 9, 82]
decap_list = [0, 0, 2, 1, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 2, 4, 0, 2, 0, 2, 0, 6, 1, 0, 8, 3, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 1, 1]
connect_port_list = [x+1 for x,y in enumerate(decap_list) if y != 0]


decap_num_list = [x-1 for x in decap_list if x != 0]


#decap_num_list    = [x-1 for x in [9,1,4,5,1,2,2,2,2,4,3,7,2,2,1]] #1,3,4,6,7,8,9,10,11,12,13,14,15,16,18
R = 0.012
Zmax = 0.026


z_orig = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['z']
brd_shape_ic = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['brd_shape_ic']
ic_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['ic_xy_indx']
top_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['top_decap_xy_indx']
bot_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['bot_decap_xy_indx']
stackup = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['stackup']
die_t = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['die_t']
sxy = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['sxy']
ic_via_xy = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['ic_via_xy']
vrm_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['vrm_xy_indx']
vrm_loc = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['vrm_loc']


top_caps = np.zeros((np.shape(top_decap_xy_indx)[0],2))
bot_caps = np.zeros((np.shape(bot_decap_xy_indx)[0],2))
for i in range(0,top_decap_xy_indx.shape[0]):
    top_caps[i] = [(top_decap_xy_indx[i,0]+.5)*200e-3/16, top_decap_xy_indx[i,1]*200e-3/16]

for i in range(0,bot_decap_xy_indx.shape[0]):
    bot_caps[i] = [(bot_decap_xy_indx[i,0]+.5)*200e-3/16, bot_decap_xy_indx[i,1]*200e-3/16]
#JT.save_2_mat([top_caps,bot_caps], ["TopDecaps", "BotDecaps"], "Case 4 - 100 Cap Test Decap Coords")

ann_size = 15
leg_size = 14

'''Save S-parameter'''
brd.save2s(z_orig, str(n))

'''Plot board shape, etc.'''
fig, ax = plt.subplots()
ax.plot(sxy[:,2], sxy[:,3])
#x.plot(ic_via_xy[:,0], ic_via_xy[:,1], 'ro')
ax.plot(ic_via_xy[0:9,0], ic_via_xy[0:9,1], 'ro')  # IC PWR
ax.plot(ic_via_xy[9:21,0], ic_via_xy[9:21,1], 'o', color = 'green') # IC GND
ax.plot((top_decap_xy_indx[:,0]+0.5)*200e-3/16, (top_decap_xy_indx[:,1])*200e-3/16, 'b*')
ax.plot((bot_decap_xy_indx[:,0]+0.5)*200e-3/16, (bot_decap_xy_indx[:,1])*200e-3/16, 'g+')
ax.plot((vrm_xy_indx[0]+0.5)*200e-3/16, (vrm_xy_indx[1])*200e-3/16, 'v')
ax.xaxis.set_label_text("Length (m)", fontsize = 30)
ax.yaxis.set_label_text("Length (m)", fontsize = 30)

'''Annotate Port Locations'''

for i in range(0, top_decap_xy_indx.shape[0]):
    ax.annotate(str(i+1), xy=((top_decap_xy_indx[i,0]+0.7)*200e-3/16, top_decap_xy_indx[i,1]*200e-3/16), size = 24)
    
for j in range(0, bot_decap_xy_indx.shape[0]):
    ax.annotate(str(j+1+top_decap_xy_indx.shape[0]), xy=((bot_decap_xy_indx[j,0]+0.7)*200e-3/16, bot_decap_xy_indx[j,1]*200e-3/16), size = 24)
    
if vrm_loc == 0:
    ax.annotate('VRM (Bottom)', xy=((vrm_xy_indx[0]+0.7)*200e-3/16, (vrm_xy_indx[1])*200e-3/16), size =24)
else:
    ax.annotate('VRM (Top)', xy=((vrm_xy_indx[0]+0.7)*200e-3/16, (vrm_xy_indx[1])*200e-3/16), size = 24)
    
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 200e-3)
ax.set_ylim(0, 200e-3)

ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)

plt.legend(['Board', 'IC PWR Vias', 'IC GND Vias', 'Top decaps', 'Bottom decaps'], fontsize = 17)

fig = plt.gcf()
fig.set_size_inches((9, 12), forward=False)
#plt.savefig('Case 1', bbox_inches='tight')

plt.show()

'''Plot board stackup'''
die_t = die_t.round(decimals=4)
die_t_reverse = list(reversed(list(die_t * 1e3)))
stackup_reverse = list(reversed(list(stackup)))

die_t_string = list(map(str, reversed(list(die_t*1e3))))
df = pd.DataFrame([reversed(list(die_t*1e3))], columns=die_t_string, index=['Stackup'])
ax = df.plot(kind='bar', stacked=True, rot=360, legend='reverse', fontsize = 20)
ax.legend(fontsize = 30)
ax.xaxis.set_label_text("")
ax.yaxis.set_label_text("Thickness (mm)", fontsize = 30)



for i in range(0, len(stackup_reverse)):
    if i == 0:
        ax.annotate('GND',(-0.2, 0), size= 20)
    elif int(stackup_reverse[i]) == 1:
        ax.annotate('PWR',(-0.2, np.sum(die_t[-i:]*1e3)-0.01), size = 30)
    elif int(stackup_reverse[i]) == 0:
        ax.annotate('GND',(-0.2, np.sum(die_t[-i:]*1e3) - 0.01), size = 30)

ax.tick_params(axis="x", labelsize=30)
ax.tick_params(axis="y", labelsize=30)
#plt.savefig('Case 1 Stackup', bbox_inches='tight')
plt.show()




'''Connect decaps'''
z_with_decap, _ = brd.connect_n_decap(z_orig,
                                      map2orig_input = list(range(0, z_orig.shape[1])),
                                      connect_port_list = connect_port_list,
                                      decap_num_list    = decap_num_list
                                      )


plt.loglog(brd.freq.f[0:132], np.abs(z_with_decap[0:132,0,0]))
plt.loglog(brd.freq.f[0:132], get_target_z_RL(R, Zmax), 'r--')
plt.grid(which='both')
plt.xlabel('Frequency(Hz)', size = 28)
plt.ylabel('Impedance(Ohm)', size = 28)
plt.title('Impedance Curve for Capacitor Placement', fontsize = 28)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=ann_size)
plt.legend(['Solution Impedance','Allowable Impedance'], fontsize = 24)
plt.show()


'''Plot decap placement'''
fig, ax = plt.subplots()
ax.plot(sxy[:,2], sxy[:,3])
ax.plot(ic_via_xy[0:9,0], ic_via_xy[0:9,1], 'ro')  # IC PWR
ax.plot(ic_via_xy[9:21,0], ic_via_xy[9:21,1], 'o', color = 'green') # IC GND
ax.plot((top_decap_xy_indx[:,0]+0.5)*200e-3/16, (top_decap_xy_indx[:,1])*200e-3/16, 'b*')
ax.plot((bot_decap_xy_indx[:,0]+0.5)*200e-3/16, (bot_decap_xy_indx[:,1])*200e-3/16, 'g+')
ax.plot((vrm_xy_indx[0]+0.5)*200e-3/16, (vrm_xy_indx[1])*200e-3/16, 'v')



"Annotate Based on Decap Type"
for i in range(0, len(connect_port_list)):
    if connect_port_list[i] <= top_decap_xy_indx.shape[0]:
        ax.annotate(str(decap_num_list[i]+1), xy=((top_decap_xy_indx[connect_port_list[i]-1,0]+0.7)*200e-3/16,
                                                top_decap_xy_indx[connect_port_list[i]-1,1]*200e-3/16),
                                color = 'red', size = 28)

    else:
        ax.annotate(str(decap_num_list[i]+1), xy=((bot_decap_xy_indx[connect_port_list[i]-top_decap_xy_indx.shape[0]-1,0]+0.7)*200e-3/16,
                                                bot_decap_xy_indx[connect_port_list[i]-top_decap_xy_indx.shape[0]-1,1]*200e-3/16),
                    color='red', size = 28)


#"Annotate Based on Port Priority"
# Requires a Port Priority Sequence"
# connect_port_list = np.copy(prio)
# for i in range(0, len(connect_port_list)):
#     if connect_port_list[i] <= top_decap_xy_indx.shape[0]:
#         ax.annotate(str(i + 1),
#                     xy=((top_decap_xy_indx[connect_port_list[i] - 1, 0] + 0.7) * 200e-3 / 16,
#                         top_decap_xy_indx[connect_port_list[i] - 1, 1] * 200e-3 / 16),
#                     color='red', size = 28)
#     else:
#         ax.annotate(str(i + 1), xy=(
#         (bot_decap_xy_indx[connect_port_list[i] - top_decap_xy_indx.shape[0] - 1, 0] + 0.7) * 200e-3 / 16,
#         bot_decap_xy_indx[connect_port_list[i] - top_decap_xy_indx.shape[0] - 1, 1] * 200e-3 / 16),
#                     color='red', size=28)
#
# if vrm_loc == 0:
#     ax.annotate('VRM (Bottom)', xy=((vrm_xy_indx[0]+0.7)*200e-3/16, (vrm_xy_indx[1])*200e-3/16), size = 28)
# else:
#     ax.annotate('VRM (Top)', xy=((vrm_xy_indx[0]+0.7)*200e-3/16, (vrm_xy_indx[1])*200e-3/16), size = 28)
#

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 200e-3)
ax.set_ylim(0, 200e-3)
ax.xaxis.set_label_text("Length (m)", fontsize = 30)
ax.yaxis.set_label_text("Length (m)", fontsize = 30)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)
plt.title('New Scoring + Shuffle', fontsize = 30)
#plt.title('Full Port Priority for 100 Port Board CCW', fontsize = 30)
plt.legend(['Board', 'IC PWR Vias', 'IC GND Vias', 'Top decaps', 'Bottom decaps'], fontsize = 17.5)
plt.show()

