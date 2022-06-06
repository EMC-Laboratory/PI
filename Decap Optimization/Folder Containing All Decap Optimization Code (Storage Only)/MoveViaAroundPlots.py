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



# Read from file
save_path = 'Testing for Moving Vias/'
file_name = '100 mm Square 2mm Segments.npz'
save_file_path = os.path.join(save_path, file_name)

# sxy = np.load(os.path.join(save_file_path +'.npz'))['sxy']
# L_self = np.load(os.path.join(save_file_path +'.npz'))['L']
# coords = np.load(os.path.join(save_file_path +'.npz'))['coords']
# dist_border = np.load(os.path.join(save_file_path +'.npz'))['dist_from_border']
# dist_via = np.load(os.path.join(save_file_path +'.npz'))['via_clearance']
# print('# of Vias', len(coords))
# print('Minimum distance from VERTICES of the border (mm)', dist_border)
# print('Distance between Vias (mm)', dist_via)
# plt.scatter(coords[:,0],coords[:,1], marker = 's', c = L_self*1e6, cmap= 'binary')
# plt.plot(sxy[:,2],sxy[:,3])
# cbar = plt.colorbar()
# cbar.set_label('PUL Inductance in uH', fontsize = '20', rotation = 90)
# cbar.ax.tick_params(labelsize = '18')
# plt.title('Color Map of Self Inductance, Multiplying Line Segment Points by 5', fontsize = '24')
# plt.xlabel('Length in mm', fontsize = '24')
# plt.ylabel('Length in mm', fontsize = '24')
# plt.xticks(fontsize = '16')
# plt.yticks(fontsize = '16')
# plt.show()

#Mutual Stuff
save_path = 'Testing for Moving Vias/'
file_name = 'New Boundary Mutual'
save_file_path = os.path.join(save_path, file_name)

sxy = np.load(os.path.join(save_file_path +'.npz'))['sxy']
L = np.load(os.path.join(save_file_path +'.npz'))['L']
coords = np.load(os.path.join(save_file_path +'.npz'))['coords']
dist_border = np.load(os.path.join(save_file_path +'.npz'))['dist_from_border']
dist_via = np.load(os.path.join(save_file_path +'.npz'))['via_clearance']

per_err = np.zeros((len(coords)-1))
mut_L = np.zeros((len(coords)-1))

for i in range(0,len(coords)-1):
    per_err[i] = abs(abs(L[i][0,1] - L[i][1,0])/( (L[i][0,1] + L[i][1,0])/2) * 100)
for i in range(0,len(coords)-1):
    mut_L[i] = L[i][0,1]
print('Min Per Err', min(per_err))
print('Average Per Err', np.average(per_err))
print('# of Vias', len(coords))
print('Minimum distance from VERTICES of the border (mm)', dist_border)
print('Distance between Vias (mm)', dist_via)

plt.scatter(coords[0,0],coords[0,1], marker= 'x', color = 'r', s= [200])
#plt.legend(['Fixed Reference Via at {}'.format(coords[0])], fontsize = '20')
plt.scatter(coords[1::,0],coords[1::,1], marker = 's', c = per_err, cmap= 'binary')
#plt.scatter(coords[1::,0],coords[1::,1], marker = 's', c = mut_L * 1e6, cmap= 'binary')
plt.plot(sxy[:,2],sxy[:,3])
cbar = plt.colorbar()
cbar.set_label('Percent Difference', fontsize = '20', rotation = 90)
#cbar.set_label('Mutual Inductance in uH', fontsize = '20', rotation = 90)
cbar.ax.tick_params(labelsize = '18')
plt.title('Color Map of Percent Difference of Diagonals', fontsize = '24')
#plt.title('Color Map of Mutual Inductance Lref,i, Multiplying Line Seg Points by 5x', fontsize = '24')
plt.xlabel('Length in m', fontsize = '24')
plt.ylabel('Length in m', fontsize = '24')
plt.xticks(fontsize = '16')
plt.yticks(fontsize = '16')
plt.show()