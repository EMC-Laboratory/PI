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
import matplotlib.path as path
import random

#### Initialize Board ###
brd = PDN()
BASE_PATH = 'new_data_test/'

########## Remake info from a npz file ########
# read info from npz file
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

# Set BEM settings
brd.outer_bd_node = np.array([[0,0], [100,0], [100,100], [0,100], [0,0]]) *1e-3 # default boundary.
                                                                    # Only to be able to initialize a board
brd.seg_len = 2e-3
brd.er = 4.3
brd.ic_via_xy = np.array([[100,100]]) * 1e-3 # default via location
brd.ic_via_type = np.array([1])
brd.stackup = np.copy(stackup)
brd.die_t = np.copy(die_t)
brd.via_r = 0.1e-3
via_dist = 1e-3
max_len = 200e-3
img_size = 16
unit_size = max_len/16
e = 8.85e-12
brd.init_para()

# Overwrite initial board settings with new shape and fix associated values

BASE_PATH = 'Testing for Moving Vias/'

########## Remake info from a npz file ########
# read info from npz file

brd.sxy = np.load(os.path.join(BASE_PATH, "100 mm Square 2mm Segments"+'.npz'))['sxy']
print(np.shape(brd.sxy), 'HERE')
#brd.sxy = 5 * np.copy(sxy) # overwrite default segmented boundary with the segmented boundary from npz file
brd.area = PolyArea(sxy[:,0],sxy[:,1]) # adjust area after changing boundary
brd.C_pul = brd.er * e * brd.area / 1  # recalculate C PUL for new area

# Change via locations
holder = np.concatenate((brd.sxy[:,2:4], np.reshape(brd.sxy[0,0:2],(1,2))), axis= 0)
# print(sxy[:,2:4],'\n')
# print(holder,'\n')
# print(sxy[0,0:2])

curve_path = path.Path(holder)


#### generate random coords

# Get list of coords to test
num_points = 301
mutual_L = True


#Array to hold coordinates
coords = np.zeros((num_points,2))
ref_via = np.array([50e-3, 50e-3])  # ref via for mutual inductance stuff

dist_from_border = 3e-3
dist_from_border2 = 10e-3
min_via_clearance = 3e-3


# the check for distance between vias assumes that an x coordinate of 0 is not possible
# for i in range(num_points):
#     clearance_flag = False
#     point_true = False
#     if mutual_L and i == 0:
#         print('Insert', ref_via, 'into first index as reference for mutual inductance')
#         coords[i] = np.copy(ref_via)
#
#     else:
#         while not point_true:
#             x_coord = random.random() * 200e-3
#             y_coord = random.random() * 200e-3
#
#             #check distance
#             distance_array = [math.sqrt( (holder[j,0] - x_coord)**2 + (holder[j,1] - y_coord)**2) for j in range(len(holder))]
#             if np.any(coords):
#                 distance_array2 = [math.sqrt( (coords[j,0] - x_coord)**2 + (coords[j,1] - y_coord)**2) for j in range(np.count_nonzero(coords[:,1]))]
#                 if min(distance_array2) > min_via_clearance:
#                     clearance_flag = True
#                 else:
#                     clearance_flag = False # for some reason, I need to explictily make it false or else this won't work
#             else:
#                 clearance_flag = True
#
#             if min(distance_array) > dist_from_border and curve_path.contains_point((x_coord,y_coord)):
#             #if min(distance_array) < dist_from_border and curve_path.contains_point((x_coord, y_coord)) and min(distance_array) > dist_from_border2:
#                 if clearance_flag:
#                     point_true = True
#
#         coords[i] = np.array([x_coord,y_coord])

coords = np.load(os.path.join(BASE_PATH, "100 mm Square 50mm Segments Convergence"+'.npz'))['coords']


#### after generating coordinates, calc out all the L's

#  L matrix holder

# for self
L = np.zeros((num_points))



# for mutual
L = np.zeros((num_points-1,2,2))

if mutual_L:
    for i in range(1,num_points):
        brd.ic_via_xy = np.concatenate((np.copy(np.reshape(coords[0],(1,2))),
                                       np.copy(np.reshape(coords[i],(1,2))))) # new via position
        brd.via_xy = np.copy(brd.ic_via_xy)
        print('Calc Mat for Via {}'.format(i+1))
        brd.calc_mat_wo_decap()
        Lpul = brd.L_pul
        L[i-1] = Lpul
        print(L[i-1],'here')

else:
    for i in range(num_points):
        brd.ic_via_xy = np.copy(np.reshape(coords[i],(1,2))) # new via position
        brd.via_xy = np.copy(brd.ic_via_xy)
        print('Calc Mat for Via {}'.format(i+1))
        brd.calc_mat_wo_decap()
        Lpul = brd.L_pul
        L[i] = Lpul[0][0]

save_path = 'Testing for Moving Vias/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
file_name = '100 mm Square 2mm Segments Convergence'
save_file_path = os.path.join(save_path, file_name)
np.savez(save_file_path, sxy = brd.sxy, L = L, coords = coords, dist_from_border = dist_from_border, via_clearance = min_via_clearance)


plt.plot(brd.sxy[:,2], brd.sxy[:,3])
plt.scatter(coords[:,0],coords[:,1])
plt.show()
