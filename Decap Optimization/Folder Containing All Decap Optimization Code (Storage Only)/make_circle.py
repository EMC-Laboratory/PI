import numpy as np
import math
from pdn_class import PDN
import os
from config2 import Config
import ShapePDN as pdn1
import copy
import math
from pdn_class2 import *
import matplotlib.pyplot as plt

a = 10 * 100e-3
h = 2e-3


radial_distances = [10 * 90e-3 * (i-1)/40 for i in range(1,41)]
angles = [-4 * math.pi * (1 - i)/40 for i in range(1,41)] # angle between from port 1 to the jth port

x = [radial_distances[i] * math.cos(angles[i]) for i in range(0,40)]
y = [radial_distances[i] * math.sin(angles[i]) + a for i in range(0,40)]
via_locations = [ [x[i],y[i]] for i in range(40)]
plt.scatter(x,y)

num_pts = 360
circ_angles = np.linspace(0,360,360) * math.pi/180
circ_x = [a * math.cos(i) for i in circ_angles]
circ_y = [a * math.sin(i) + a for i in circ_angles]
radii = [ math.sqrt(math.pow(circ_x[i],2) + math.pow(circ_y[i],2)) for i in range(0,40)]
circle_boundary = [ [circ_x[i], circ_y[i]] for i in range(num_pts)]
circle_boundary[-1][0] = circle_boundary[0][0]
circle_boundary[-1][1] = circle_boundary[0][1]
print(circle_boundary)
plt.scatter(circ_x,circ_y)
plt.xlabel('Length in m', fontsize = '20')
plt.ylabel('Length in m', fontsize = '20')
plt.xticks(fontsize = '16')
plt.yticks(fontsize = '16')
plt.title('Plate and Port Locations, c = 0.02', fontsize = '20')
plt.legend(['Via Locations', 'Boundary'], fontsize = '20')
plt.axis('equal')
plt.show()


brd = PDN()
brd.outer_bd_node = np.array(circle_boundary)
brd.seg_len = 6e-3 
brd.er = 4.3
brd.ic_via_xy = np.array(via_locations)
brd.ic_via_type = np.array([1 for i in range(40)])
brd.decap_via_xy = np.array([[1,0]])
brd.decap_via_type = np.array([1,0])
brd.decap_via_loc = np.array([1,1])
brd.stackup = np.array([1,0])
brd.die_t = np.array([2]) * 1e-3
brd.via_r = 1e-3
brd.init_para()


print('Begin Calculating Z')
brd.calc_mat_wo_decap()
L = brd.L_pul
print(np.shape(L))


# Check if BEM L matches Analytical Solution L

# Self L
u = 4*pi*1e-7
analytical_self_L = [ h * 1e9 * -u/(2*pi) * (math.log( brd.via_r * (a*a - i*i)/math.pow(a,3))
                      - (i*i)/(a*a) + 3/4) for i in radial_distances]
plt.plot(list(range(1,41)), analytical_self_L , '-o')
plt.plot(list(range(1,41)), np.diag(L) * h * 1e9)
plt.xlabel('Port Number', fontsize = '20')
plt.ylabel('Inductance in nH', fontsize = '20')
plt.xticks(fontsize = '16')
plt.yticks(fontsize = '16')
plt.legend(['Analytical Solution', ' BEM Solution'], fontsize = '20')
plt.title('Self Inductance of the ith Port, c = 0.02', fontsize = '20')
plt.grid(which= 'both')
plt.show()

# # Mutual L
# # Verifying L1i
# #angles = [4 * math.pi * (i - 40)/40 for i in range(1,41)] # angle from port i to port 40
# a_mut_L = [ -u/(2*pi) * 1e9 * h *(
#     (1/2 * math.log((radial_distances[0]**2 + radial_distances[i]**2 - 2 * radial_distances[0] * radial_distances[i] * math.cos(angles[i]))/(a*a)))
#      - (radial_distances[0]**2 + radial_distances[i]**2) / (2*a*a) +
#      1/2 * math.log( (radial_distances[0]**2)*(radial_distances[i]**2)/(math.pow(a,4)) + 1
#                      - 2 * radial_distances[0]*radial_distances[i]/(a*a)) + 3/4) for i in range(1,40)]
# #mut_L_BEM = L[39,0:39]
# #print(mut_L_BEM)
# # print(L[39])
# mut_L_BEM = L[0,1::]
# plt.plot(list(range(2,41)), a_mut_L, '-o')
# plt.plot(list(range(2,41)), mut_L_BEM * h * 1e9)
# #plt.plot(list(range(1,40)), a_mut_L, '-o')
# #plt.plot(list(range(1,40)), mut_L_BEM * h/2 * 1e9)
# plt.xlabel('Port Number')
# plt.ylabel('Inductance in nH')
# plt.legend(['BEM Solution'])
# plt.title('BEM: Mutual Inductance Between Port 1 and the ith Port (Height = 2mm)')
# plt.grid(which= 'both')
# plt.show()
#
