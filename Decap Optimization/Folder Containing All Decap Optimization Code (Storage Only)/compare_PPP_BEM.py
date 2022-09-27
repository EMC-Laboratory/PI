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
import scipy.io

##### Read BEM and PPP#####
L = np.load('20 Port L Mat 10200.npz')['L'] # From BEM
data = np.load('20 Port L Mat 10200.npz')
Lppp = scipy.io.loadmat('20 Decap Matrix.mat') # From PPP
sxy = np.load('20 Port L Mat 10200.npz')['sxy']

num_IC = 21
num_pwr = 9
num_decap_ports = 20
num_vias = num_IC + num_decap_ports * 2

# delete out ground pins for BEM
del_ports = [9 + i for i in range(num_IC - num_pwr)] + [ (num_IC-1) + 2*j for j in range(1,num_decap_ports+1)]
del_ports.reverse()

for i in del_ports:
    L = np.delete(np.delete(L,i,1),i,0)

#Get self Inductance of vias
L_self = np.diag(L)
L_from_ppp = np.diag(Lppp['x']) * 1e-12 / (0.5e-3 + 2 * 0.03429e-3)
print(L_self)
print(L_from_ppp)
# plt.plot(range(1,30), L_self)
# plt.plot(range(1,30), L_from_ppp)
# plt.show()

#plt.plot(sxy[:,2], sxy[:,3])
#plt.show()