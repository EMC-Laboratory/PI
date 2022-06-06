from pdn_class import PDN
import numpy as np
import os
import ShapePDN as pdn1
import copy
import numpy as np
import math
from pdn_class2 import *
import matplotlib.pyplot as plt


#### Initialize Board
brd = PDN()

BASE_PATH = 'Testing for Moving Vias/'
bd = np.load(os.path.join(BASE_PATH, "Recreated 100 Port Boundary"+'.npz'))['bd']
sxy_ref = np.load(os.path.join(BASE_PATH, "1"+".npz"))['sxy']
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)



#Give points that make the outer boundary. must be given CCW is believe
brd.outer_bd_node = np.array([[0,0],[100,0],[100,100],[0,100],[0,0]]) *1e-3

# minimum length between points that make up boundary
brd.seg_len = 2e-3
brd.er = 4.3

#xy locations for ic vias
brd.ic_via_xy = np.array([[25,25],[50,25],[75,25],[25,50],[50,50],[75,50],[25,75],[50,75],[75,75]])*1e-3

# type of vias. 1 for pwr IC via, 0 for gnd IC via. Corresponds to brd.ic_via_xy
brd.ic_via_type = np.array([1,0,1,0,1,0,1,0,1])

# capacitor locations
# brd.decap_via_xy = np.array([[26,8],[28,8],[27,15],[27,13],[13,25],[13,27],[21,25],[21,27],[18,21],[18,23],[14,20],[14,22]
#                              ])*1e-3
# Same as IC via type. Denote 1 as pwr via of decap, 0 for ground. length should match brd.decap_via_xy
# brd.decap_via_type = np.array([1,0,1,0,1,0,1,0,1,0,1,0])
# Vector to described the location of the via of the decap. Assuming two terminal capacitors, the entire capacitor
# will sit either on the top or the bottom of the board. So each pair of P-G vias for each decap should have the same location
# 0 indicates on bottom side of board. 1 for top side of board.
# brd.decap_via_loc = np.array([0,0,1,1,1,1,0,0,0,0,1,1])

# Brd stackup. 0 indicates GND layer, 1 indicates PWR layer
brd.stackup = np.array([0,1])

# thickness of dielectrics between layers
brd.die_t = np.array([0.2]) * 1e-3

# radius of vias
brd.via_r = 0.1e-3
brd.init_para()

##O Use to calculate z parameter considering only IC vias
brd.calc_mat_wo_decap()

L = brd.L_pul
Gh = brd.Gh
D = brd.D
np.savez('100mm Square Test', sxy = brd.sxy, L = L, ic = brd.ic_via_xy, Gh = Gh, D = D)

plt.plot(brd.sxy[:,2],brd.sxy[:,3])
plt.scatter(brd.ic_via_xy[:,0],brd.ic_via_xy[:,1])

plt.show()