from pdn_class import PDN
import numpy as np
import os
from config2 import Config
import ShapePDN as pdn1
import copy
import numpy as np
import math
from pdn_class2 import *
import matplotlib.pyplot as plt

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

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

opt = OptionsInit()                             # Initialize Run Settings
cap_objs, cap_objs_z = decap_objects(opt)


#### Initialize Board
brd = PDN()


BASE_PATH = 'Testing for Moving Vias/'
bd = np.load(os.path.join(BASE_PATH, "Recreated 100 Port Boundary"+'.npz'))['bd']
sxy_ref = np.load(os.path.join(BASE_PATH, "1"+".npz"))['sxy']
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

bd_x = bd[0::2]
bd_y = bd[1::2]

bd_x = np.append(bd_x,bd[0])
bd_x[1:np.shape(bd_x)[0]-1] = np.flip(bd_x[1:np.shape(bd_x)[0]-1])



bd_y = np.append(bd_y,bd[1])
bd_y[1:np.shape(bd_y)[0]-1] = np.flip(bd_y[1:np.shape(bd_y)[0]-1])

outer_bd = [ [bd_x[i], bd_y[i]] for i in range(len(bd_x))]

BASE_PATH = 'new_data_test'
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
vrm_xy_indx = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['vrm_xy_indx']
vrm_loc = np.load(os.path.join(BASE_PATH, str(n)+'.npz'))['vrm_loc']
unit_size = 200e-3 / 16
via_dist = 1e-3
e = 8.85e-12

#brd.outer_bd_node = np.array([[0,0],[100,0],[100,100],[0,100],[0,0]]) *1e-3
brd.outer_bd_node = np.array(outer_bd)
brd.seg_len = 2e-3
brd.er = 4.3
brd.ic_via_xy = np.copy(ic_via_xy)
brd.ic_via_type = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
brd.stackup = np.copy(stackup)
brd.die_t = np.copy(die_t)
brd.via_r = 0.1e-3

# Convert indices of vias to xy coords. Also create ground vias.
# +1 to the shapes to add in VRM via
brd.decap_via_xy = np.zeros(((top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] +1) * 2, 2))
brd.decap_via_type = np.zeros(((top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] +1) * 2))
brd.decap_via_loc = np.zeros(((top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] +1) * 2))
for i in range(0, top_decap_xy_indx.shape[0]):
    brd.decap_via_xy[2 * i, 0] = (top_decap_xy_indx[i, 0] + 0.5) * unit_size - via_dist / 2
    brd.decap_via_xy[2 * i, 1] = (top_decap_xy_indx[i, 1] + 0.5) * unit_size
    brd.decap_via_xy[2 * i + 1, 0] = (top_decap_xy_indx[i, 0] + 0.5) * unit_size + via_dist / 2
    brd.decap_via_xy[2 * i + 1, 1] = (top_decap_xy_indx[i, 1] + 0.5) * unit_size
    brd.decap_via_type[i * 2] = 1
    brd.decap_via_type[i * 2 + 1] = 0
    brd.decap_via_loc[i * 2] = 1
    brd.decap_via_loc[i * 2 + 1] = 1
for j in range(0, bot_decap_xy_indx.shape[0]):
    k = j + top_decap_xy_indx.shape[0]
    brd.decap_via_xy[2 * k, 0] = (bot_decap_xy_indx[j, 0] + 0.5) * unit_size - via_dist / 2
    brd.decap_via_xy[2 * k, 1] = (bot_decap_xy_indx[j, 1] + 0.5) * unit_size
    brd.decap_via_xy[2 * k + 1, 0] = (bot_decap_xy_indx[j, 0] + 0.5) * unit_size + via_dist / 2
    brd.decap_via_xy[2 * k + 1, 1] = (bot_decap_xy_indx[j, 1] + 0.5) * unit_size
    brd.decap_via_type[k * 2] = 1
    brd.decap_via_type[k * 2 + 1] = 0
    brd.decap_via_loc[k * 2] = 0
    brd.decap_via_loc[k * 2 + 1] = 0
# Convert indicies for the via where VRM goes
brd.decap_via_xy[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1) * 2 - 2, 0] = (vrm_xy_indx[
                                                                                                  0] + 0.5) * unit_size - via_dist / 2
brd.decap_via_xy[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1) * 2 - 2, 1] = (vrm_xy_indx[
                                                                                                  1] + 0.5) * unit_size
brd.decap_via_xy[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1) * 2 - 1, 0] = (vrm_xy_indx[
                                                                                                  0] + 0.5) * unit_size + via_dist / 2
brd.decap_via_xy[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1) * 2 - 1, 1] = (vrm_xy_indx[
                                                                                                  1] + 0.5) * unit_size
brd.decap_via_type[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1) * 2 - 2] = 1
brd.decap_via_type[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1) * 2 - 1] = 0
brd.decap_via_loc[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1) * 2 - 2] = vrm_loc
brd.decap_via_loc[(top_decap_xy_indx.shape[0] + bot_decap_xy_indx.shape[0] + 1) * 2 - 1] = vrm_loc

brd.init_para()




print('Calculating Z')
z = brd.calc_z()
L = brd.L_pul
print(np.shape(z),'shape z')
#add VRM to s parameters

freq = np.logspace(np.log10(brd.fstart), np.log10(brd.fstop), brd.nf)
R = 3e-3
jwL = 2j *math.pi * 2.5e-9 * freq
RL = R + jwL
RL = np.reshape(RL, (RL.shape[0],1,1))
z_with_vrm = pdn1.new_connect_1decap(z, 101, RL)


brd.save2s(z_with_vrm, '100 Port Board CCW S-Parameters')

#### Plotting only, doesn't matter ####
x =  (top_decap_xy_indx + 0.5) * 200e-3/16
#x = np.concatenate((x, (top_decap_xy_indx - 0.5) * 200e-3/16))
y =  (bot_decap_xy_indx + 0.5)* 200e-3/16

decap_via_xy = np.concatenate((x,y))
# decap_via_type = np.resize(np.array([1]), (len(x) + len(y)))
# decap_via_loc = np.concatenate((np.ones(len(x)), np.zeros(len(y))))

plt.plot(brd.sxy[:,2], brd.sxy[:,3])
plt.scatter(brd.via_xy[:9,0], brd.via_xy[:9,1], color = 'red')
plt.scatter(brd.via_xy[9:len(ic_via_xy),0], brd.via_xy[9:len(ic_via_xy),1], color = 'green')
plt.scatter(decap_via_xy[0: len(top_decap_xy_indx) + 1,0], decap_via_xy[0:len(top_decap_xy_indx)+1,1], marker = '*', color = 'blue')
plt.scatter(decap_via_xy[len(top_decap_xy_indx)::,0], decap_via_xy[len(top_decap_xy_indx)::,1], marker = '+', color = 'green')
#plt.scatter(brd.decap_via_xy[-1,0], brd.decap_via_xy[-1,1], marker = 'v', color = 'red')
plt.scatter(brd.decap_via_xy[-1,0], brd.decap_via_xy[-1,1], marker = 'v', color = 'red')

plt.title('Recreated Boundary 5x Size', fontsize = 24)
plt.xlabel('Length in m', fontsize = 24)
plt.ylabel('Length in m', fontsize = 24)
plt.legend(['Board Shape', 'IC Power Vias', 'IC Ground Vias', 'Top Decaps', 'Bottom Decaps'], loc = 'upper left')
plt.show()

### Plot impedance
plt.loglog(freq, np.abs(z_with_vrm[:,0,0]))
plt.grid(which='both')
plt.show()