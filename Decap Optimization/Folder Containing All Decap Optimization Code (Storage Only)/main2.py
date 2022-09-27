# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:28:42 2020

@author: lingzhang0319
"""

from pdn_class2 import *
import numpy as np
from math import sqrt, pi, sin, cos, log, atan, pi
import skrf as rf
import matplotlib.pyplot as plt




##### adding capacitors 1 at a time
brd2 = PDN()

brd2.outer_bd_node = np.array([[0,0],[5,0],[5,5],[10,5],[10,0],[20,0],[20,5],[30,5],[30,0],[40,0],
                              [40,8],[25,20],[25,30],[10,30],[10,15],[0,10],[0,0]])*1e-3
brd2.seg_len = 2e-3
brd2.er = 4.3
#brd2.seg_bd()
brd2.ic_via_xy = np.array([[10,10],[12,10]])*1e-3
brd2.ic_via_type = np.array([1,0])
brd2.stackup = np.array([0,0,1,0])
brd2.die_t = np.array([0.3e-3,0.3e-3,2e-3])
brd2.via_r = 0.1e-3

brd2.init_para()

brd2.calc_mat_wo_decap()
brd2.add_decap(26e-3,8e-3,28e-3,8e-3,9,0)
brd2.add_decap(27e-3,15e-3,27e-3,13e-3,7,1)
brd2.add_decap(13e-3,25e-3,13e-3,27e-3,5,1)
brd2.add_decap(21e-3,25e-3,21e-3,27e-3,3,0)
brd2.add_decap(18e-3,21e-3,18e-3,23e-3,5,0)


#### adding all the capacitors at once
brd = PDN()

brd.outer_bd_node = np.array([[0,0],[5,0],[5,5],[10,5],[10,0],[20,0],[20,5],[30,5],[30,0],[40,0],
                              [40,8],[25,20],[25,30],[10,30],[10,15],[0,10],[0,0]])*1e-3
brd.seg_len = 2e-3
brd.er = 4.3
#brd.seg_bd()
brd.ic_via_xy = np.array([[10,10],[12,10]])*1e-3
brd.ic_via_type = np.array([1,0])
brd.stackup = np.array([0,0,1,0])
brd.die_t = np.array([0.3e-3,0.3e-3,2e-3])
brd.via_r = 0.1e-3


## Coordinates for [topX, topY], [bottomX, bottomY], first entry gives top plane position. Second gives bottom plane
# position
brd.decap_via_xy = np.array([[26,8],[28,8],[27,15],[27,13],[13,25],[13,27],[21,25],[21,27],[18,21],[18,23]])*1e-3

# Same here, via type for each top bottom coordinate
brd.decap_via_type = np.array([1,0,1,0,1,0,1,0,1,0])

# WHere the capacitor sits, for each via location. Since capacitor will sit on one layer, regardless of via types
# the position will be the same ie [0,0], [1,1]
brd.decap_via_loc = np.array([0,0,1,1,1,1,0,0,0,0])

brd.init_para()

z = brd.calc_z()   # returns with merged IC no decaps z parameters

# connect with decaps
brd_z,_ = brd.connect_n_decap(z,list(range(0,len(z))),[1,2,3,4,5],[9,7,5,3,5])
brd.save2s(z,'python_dicrectly_calculate_z')
brd.save2s(brd_z,'test')


plt.loglog(brd2.freq.f,np.abs(brd2.z_mergeIC_with_decap[:,0,0]))
plt.loglog(brd.freq.f,np.abs(brd_z[:,0,0]),'r--',linewidth=2)
plt.grid(which='both')
plt.legend(['Method 1 (add one by one)','Method 2 (add decaps together)'])
plt.show()

