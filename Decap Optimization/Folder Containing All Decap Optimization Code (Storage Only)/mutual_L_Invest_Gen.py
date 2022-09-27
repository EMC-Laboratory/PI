import numpy as np
import os
from config2 import Config
import ShapePDN as pdn1
import copy
import matplotlib.pyplot as plt
import math
from pdn_class2 import *

BASE_PATH = 'new_data_test/'
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

n = 1
z_orig = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['z']
brd_shape_ic = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['brd_shape_ic']
ic_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['ic_xy_indx']
top_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['top_decap_xy_indx']
bot_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['bot_decap_xy_indx']


stackup = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['stackup']
die_t = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['die_t']
sxy = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['sxy']

ic_via_xy = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['ic_via_xy']
ic_via_type = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['ic_via_type']

decap_via_xy = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['decap_via_xy']
decap_via_type = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['decap_via_type']
decap_via_loc = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['decap_via_loc']

vrm_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['vrm_xy_indx']
vrm_loc = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['vrm_loc']

top_caps = np.zeros((2 * np.shape(top_decap_xy_indx)[0],2))
bot_caps = np.zeros((2 * np.shape(bot_decap_xy_indx)[0],2))
num_top = np.shape(top_decap_xy_indx)[0]
num_bot = np.shape(bot_decap_xy_indx)[0]
num_caps = num_bot + num_top


via_dist = 1e-3
for i in range(0,top_decap_xy_indx.shape[0]):
    top_caps[2*i] = [(top_decap_xy_indx[i,0]+.5)*200e-3/16, top_decap_xy_indx[i,1]*200e-3/16]
    top_caps[2 * i + 1] = [top_caps[2*i][0] + via_dist, top_caps[2*i][1]]

for i in range(0,bot_decap_xy_indx.shape[0]):
    bot_caps[2*i] = [(bot_decap_xy_indx[i,0]+.5)*200e-3/16, bot_decap_xy_indx[i,1]*200e-3/16]
    bot_caps[2 * i + 1] = [bot_caps[2 * i][0] + via_dist, bot_caps[2 * i][1]]

vrm_xy_indx = np.array((vrm_xy_indx))
vrm_spot = np.zeros((2,np.shape(vrm_xy_indx)[0]))
for i in range(0,1):
    vrm_spot[2*i] = [(vrm_xy_indx[0]+.5)*200e-3/16, vrm_xy_indx[1]*200e-3/16]
    vrm_spot[2*i + 1] = [vrm_spot[2*i][0] + via_dist, vrm_spot[2*i][1]]

def OptionsInit():
    # Get settings
    opt = Config()
    return opt

def decap_objects(opt):
    # Store capacitor library as their impedances
    cap_objs = [pdn1.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors, 1 to 10, high end is not inclusive [x,y) so need + 1
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z

opt = OptionsInit()                             # Initialize Run Settings
cap_objs, cap_objs_z = decap_objects(opt)


#### Initialize Board
brd = PDN()

brd.outer_bd_node = np.array([[0,0], [50,0], [50,50], [0,50], [0,0]]) *1e-3

brd.seg_len = 2e-3
brd.er = 4.4

brd.ic_via_xy = np.copy(ic_via_xy)
brd.ic_via_type = np.copy(ic_via_type)

brd.stackup = np.copy(stackup)
brd.die_t = np.copy(die_t)
brd.via_r = 0.1e-3

brd.decap_via_xy = np.copy(decap_via_xy)
brd.decap_via_type = np.copy(decap_via_type)
brd.decap_via_loc = np.copy(decap_via_loc)

brd.init_para()
brd.outer_sxy = np.copy(sxy)
brd.sxy = np.copy(sxy)

brd.area = PolyArea(brd.sxy[:,0],brd.sxy[:,1])
e = 8.85e-12
brd.C_pul = brd.er*e*brd.area/1

merge_ic_z = brd.calc_z()
no_merge_ic_z = np.copy(brd.z_orig)

## VRM Parameters
r_vrm = .003
l_vrm = 2.5e-9
vrm_z = [complex(r_vrm,l_vrm * i * 2 * pi ) for i in brd.freq.f]
vrm_z = r_vrm + 1j*2*pi*brd.freq.f*l_vrm
vrm_z_array = np.zeros((brd.freq.f.shape[0],1,1),dtype=complex)
vrm_z_array[:,0,0] = np.array(vrm_z)

#L = brd.L_pul * brd.die_t[0]
L = np.sum(brd.die_t) * brd.L_pul
L_test = np.save('100 Port L Mat',L)

#brd_and_vrm = pdn1.new_connect(z, opt.ic_port, vrm_z_array) # connects vrm to port 2 (index 1) of the board
#brd_and_vrm = pdn1.new_connect(z, 1, vrm_z_array)

file_name = "100 Port Merge IC"
file_name2 = "100 Port No Merge IC"

#file_name2 = "Test for L without VRM etc"

#brd.save2s(brd_and_vrm,file_name)
brd.save2s(merge_ic_z,file_name)
brd.save2s(no_merge_ic_z,file_name2)







