from data_generate import *
import numpy as np
import numpy.random as random
from scipy.special import binom
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from copy import deepcopy
from pdn_class import PDN
import os
import pandas as pd
from config2 import Config


def OptionsInit():
    # Get settings
    opt = Config()
    return opt

# BASE_PATH = 'new_data/'
BASE_PATH = 'new_data_test/'

if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

n = 1

z_orig = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['z']
brd_shape_ic = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['brd_shape_ic']
ic_xy_indx = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['ic_xy_indx']
top_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['top_decap_xy_indx']
bot_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['bot_decap_xy_indx']
stackup = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['stackup']
die_t = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['die_t']
sxy = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['sxy']
ic_via_xy = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['ic_via_xy']
vrm_xy_indx = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['vrm_xy_indx']
vrm_loc = np.load(os.path.join(BASE_PATH, str(n ) +'.npz'))['vrm_loc']

opt = OptionsInit()  # Create settings reference

L = np.load(os.path.join('Testing for Moving Vias', '100 Port Boundary Mat Test CCW' +'.npz'))['L']
#sxy = np.load('100 Port L Mat Verification Rectangle.npz')['sxy']
# for i in range(np.shape(L)[0]):
#     for j in range(np.shape(L)[1]):
#         print(L[i,j]/sum(die_t)==L2[i,j])
#         if not L[i,j]/sum(die_t)==L2[i,j]:
#             print(L[i,j]/sum(die_t))
#             print(L2[i,j])

# Extract out the self L and mutual L of between/of power pins only
num_IC = 21
num_pwr = 9
num_decap_ports = 101
num_vias = num_IC + num_decap_ports * 2
# delete out ground pins
del_ports = [9 + i for i in range(num_IC - num_pwr)] + [ (num_IC-1) + 2*j for j in range(1,num_decap_ports+1)]
del_ports.reverse()

#L = np.copy(L2)
for i in del_ports:
    L = np.delete(np.delete(L,i,1),i,0)

# for i in range(np.shape(L)[0]):
#     print(i)
#     print(L[0,i], "0,i")
#     print(L[i,0], "i,0")

#Loop shorting 1 port at a time
# port_num1 = 101
# Leq_all = np.zeros(100)
# L_vrm = 2.5e-9
# for i in range(1, num_decap_ports):
#     s1 = port_num1 + num_pwr - 1
#     port_num2 = i
#     s2 = port_num2 + num_pwr - 1
#     L_array = np.array([ [L[0,0], L[0,s1], L[0,s2]], [L[s1,0], L[s1,s1] + L_vrm, L[s1,s2]], [L[s2,0], L[s2,s1], L[s2,s2]] ])
#     B = np.linalg.inv(L_array)
#     B_reduced = np.array([[B[0,0], B[0,1] + B[0,2]], [B[1,0]+ B[2,0], B[1,1] + B[2,2] + B[1,2] + B[2,1]]])
#     Leq_array = np.linalg.inv(B_reduced)
#     Leq = Leq_array[0,0] + Leq_array[1,1] - Leq_array[1,0] - Leq_array[0,1]
#     Leq_all[i-1] = Leq
# Leq_all = Leq_all

#self inductance of vias
holder = np.delete(np.delete(L,np.flip([0,1,2,3,4,5,6,7,8]), axis = 1),np.flip([0,1,2,3,4,5,6,7,8]), axis = 0)
holder2 = np.delete(np.delete(L,np.flip([0,1,2,3,4,5,6,7,8,100]), axis = 1),np.flip([0,1,2,3,4,5,6,7,8,100]), axis = 0)
Leq_all = np.diag(L[num_pwr:-1,num_pwr:-1])

#Mutual L
# via_looking = 0
# Leq_all = L[via_looking,num_pwr:-1] * 1e9

# Mutual L percent error
# per_err = np.zeros((100))
# ref_via = 0
#
# for i in range(100):
#     print(i)
#     per_err[i] = abs(abs(L[ref_via][num_pwr + i] - L[num_pwr + i][ref_via])/( (L[ref_via][num_pwr+i] + L[num_pwr+i][ref_via])/2) * 100)

sxy = np.load(os.path.join('Testing for Moving Vias', '100 Port Boundary Mat Test CCW' + '.npz'))['sxy']

x = np.concatenate( ( (top_decap_xy_indx[:,0]+0.5)*200e-3/16, (bot_decap_xy_indx[:,0]+0.5)*200e-3/16))
y = np.concatenate( ((top_decap_xy_indx[:,1]+0.5)*200e-3/16,  (bot_decap_xy_indx[:,1]+0.5)*200e-3/16))

# caps = np.load('100 Port L Mat Increased Size.npz')['decaps']
# ic_via_xy = np.load('100 Port L Mat Increased Size.npz')['ic']
# x = np.concatenate( (caps[0:len(top_decap_xy_indx),0], caps[len(top_decap_xy_indx):-1,0]))
# y = np.concatenate( ( caps[0:len(top_decap_xy_indx),1], caps[len(top_decap_xy_indx):-1,1]))
fig, ax = plt.subplots()


ax.plot(sxy[:,2], sxy[:,3]) # plot shape
ax.plot(ic_via_xy[0:9,0], ic_via_xy[0:9,1], 'ro')  # IC PWR
ax.plot(ic_via_xy[9:21,0], ic_via_xy[9:21,1], 'o', color = 'green') # IC GND
ax.plot((vrm_xy_indx[0]+0.5)*200e-3/16, (vrm_xy_indx[1])*200e-3/16, 'v', color = 'black') # VRM

#plot inductance
plt.scatter(x, y, marker = 's', c = Leq_all/(1e3), cmap= 'viridis')
cbar = plt.colorbar()
#cbar.set_label('PUL Self Inductance in nH', fontsize='24', rotation=90)
cbar.set_label('PUL Mutual Inductance in nH', fontsize='20', rotation=90)

cbar.ax.tick_params(labelsize='20')
ax.xaxis.set_label_text("Length (m)", fontsize = 24)
ax.yaxis.set_label_text("Length (m)", fontsize = 24)
#plt.title('Color Map of Equivalent Inductance Shorting A 3rd Port', fontsize = 24)
#plt.title('Color Map of Self Inductance ', fontsize = 24)
plt.title('Color Map of Mutual Inductance to an IC Via', fontsize = 24)
plt.xticks(fontsize = '20')
plt.yticks(fontsize = '20')
plt.show()



#plot mut L error
# #plt.scatter(x, y, marker = 's', c = per_err, cmap= 'viridis')
# cbar = plt.colorbar()
# cbar.set_label('Percentage Difference', fontsize='20', rotation=90)
# cbar.ax.tick_params(labelsize='18')
# ax.xaxis.set_label_text("Length (m)", fontsize = 24)
# ax.yaxis.set_label_text("Length (m)", fontsize = 24)
# plt.title('Percentage Difference Between Mutual Terms ', fontsize = 24)
# plt.xticks(fontsize = '20')
# plt.yticks(fontsize = '20')
# #plt.legend(['Board', 'IC PWR Vias', 'IC GND Vias', 'VRM'], fontsize = 16, loc = 'upper left')
# plt.show()

