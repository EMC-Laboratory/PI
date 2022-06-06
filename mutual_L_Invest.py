from config2 import Config
import Stage1pdn as pdn1
import copy
import PopInit as Pop
import matplotlib.pyplot as plt
import numpy as np
import random
import skrf as rf
import math as math
import sys

def OptionsInit():
    # Get settings
    opt = Config()
    return opt


def decap_objects(opt):
    cap_objs = [pdn1.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z

opt = OptionsInit()  # Create settings reference
cap_objs, cap_objs_z = decap_objects(opt)  # generate capacitor objects and there z parameters

L = np.load('100 Port L Mat.npy') # 100 port + 1 actually

# Extract out the self L and mutual L of between/of power pins only
num_IC = 21
num_pwr = 9
num_decap_ports = 101
num_vias = num_IC + num_decap_ports * 2


del_ports = [9 + i for i in range(num_IC - num_pwr)] + [ (num_IC-1) + 2*j for j in range(1,num_decap_ports+1)]
del_ports.reverse()

for i in del_ports:
    L = np.delete(np.delete(L,i,1),i,0)



# With 1 shorted via, ie the VRM

#Assuming positive mutuals, in this case the closer it is the lower the Leq
port_num = 101
shorted_via = port_num + num_pwr - 1
#print(shorted_via)

obs_via = 0
Leq = L[obs_via,obs_via] + L[shorted_via,shorted_via] - L[obs_via, shorted_via] - L[shorted_via, obs_via]
#print(Leq)

#If the VRM had an extra bit of inductance to model real VRM or decap, the Leq increases for one shorted via
#Leq = L[obs_via,obs_via] + L[shorted_via,shorted_via] + VRM_L - L[obs_via, shorted_via] - L[shorted_via, obs_via]

#Assume VRM already shorted, and we short another via.
# port_num1 = 101
# s1 = port_num1 + num_pwr - 1
#
# port_num2 = 8
# s2 = port_num2 + num_pwr - 1
#
# o_via = 0
#
# L_array = np.array([ [L[0,0], L[0,s1], L[0,s2]], [L[s1,0], L[s1,s1], L[s1,s2]], [L[s2,0], L[s2,s1], L[s2,s2]] ])
# B = np.linalg.inv(L_array)
# B_reduced = np.array([[B[0,0], B[0,1] + B[0,2]], [B[1,0]+ B[2,0], B[1,1] + B[2,2] + B[1,2] + B[2,1]]])
# Leq_array = np.linalg.inv(B_reduced)
# Leq = Leq_array[0,0] + Leq_array[1,1] - Leq_array[1,0] - Leq_array[0,1]
# print(Leq)


# Compare Mutual L
port_num1 = 0
ob_mut = np.zeros(100)
port_mut = np.zeros(100)

for i in range(1, 101):
    s1 = 0
    port_num2 = i
    s2 = port_num2 + num_pwr - 1
    ob_mut[i-1] = L[s1,s2]

port_num1 = 101
for i in range(1, 101):
    s1 = port_num1 + num_pwr - 1
    port_num2 = i
    s2 = port_num2 + num_pwr - 1
    port_mut[i - 1] = L[s1,s2]

port_mut = port_mut * 1e9

ob_mut = ob_mut * 1e9

xhold1 = []
xhold2 = []
yhold1 = []
yhold2 = []

# for i in range(1,101):
#
#     if port_mut[i-1] > ob_mut[i-1]:
#         xhold1.append(i)
#         yhold1.append(i-1)
#
#     elif port_mut[i-1] < ob_mut[i-1]:
#         xhold2.append(i)
#         yhold2.append(i-1)

for i in range(1,101):


    if ob_mut[i-1] - port_mut[i-1]  < 0:
        xhold1.append(i)
        yhold1.append(i-1)
    elif ob_mut[i-1] - port_mut[i-1]  > 0:
        xhold2.append(i)
        yhold2.append(i-1)



plt.scatter(xhold1, ob_mut[np.s_[yhold1]] - port_mut[np.s_[yhold1]])
plt.scatter(xhold2, ob_mut[np.s_[yhold2]] - port_mut[np.s_[yhold2]])


#plt.scatter(bad, Leq_all[np.s_[bad]])
#plt.scatter(good, Leq_all[np.s_[good]])

of_int = [9, 82, 61, 59, 45, 21, 53, 25, 54, 60, 86, 76, 14, 36, 6, 79, 51, 22 ,35, 43]
of_int2 = of_int.copy()
of_int = [i - 1 for i in of_int]
#plt.scatter(of_int2,  ob_mut[np.s_[of_int]] - port_mut[np.s_[of_int]], color = 'black')


#plt.scatter(list(range(1,101)), ob_mut - port_mut)


print(xhold1)
print(yhold1)
print(xhold2)
print(yhold2)

plt1 = plt.figure(1)
#plt.scatter(list(range(1,101)), port_mut)
#plt.scatter(list(range(1,101)), ob_mut)



plt.xlabel('Port Num')
plt.ylabel('Mutual Inductance in nH')
#plt.title('Larger Mutual Inductance Between VRM to Via and Observing Via to Via')
plt.title('Difference of M, Observing Via to Via - VRM Via to Via')

plt.legend(['IC Mutual to Via Smaller than VRM to Via', 'IC Mutual to Via Larger than VRM to Via', 'Center Ports', 'Upper Ports'])
plt.grid(which= 'both', axis= 'both')
ax = plt.gca()
ax.xaxis.grid(True, which='minor')

#plt.show(block = True)








# Put it in a loop
port_num1 = 101
Leq_all = np.zeros(100)
for i in range(1, 101):
    s1 = port_num1 + num_pwr - 1
    port_num2 = i
    s2 = port_num2 + num_pwr - 1
    L_array = np.array([ [L[0,0], L[0,s1], L[0,s2]], [L[s1,0], L[s1,s1] + 2.5e-9, L[s1,s2]], [L[s2,0], L[s2,s1], L[s2,s2]] ])
    B = np.linalg.inv(L_array)
    B_reduced = np.array([[B[0,0], B[0,1] + B[0,2]], [B[1,0]+ B[2,0], B[1,1] + B[2,2] + B[1,2] + B[2,1]]])
    Leq_array = np.linalg.inv(B_reduced)
    Leq = Leq_array[0,0] + Leq_array[1,1] - Leq_array[1,0] - Leq_array[0,1]
    Leq_all[i-1] = Leq
Leq_all = Leq_all * 1e9



plt2 = plt.figure(2)


plt.scatter(list(range(1,101)), Leq_all)

bad = [18,47,62,10]
bad2 = [i - 1 for i in bad]
good = [21,81,7,69]
good2= [i - 1 for i in good]
segment = bad + good

of_int = [9, 82, 61, 59, 45, 21, 53, 25, 54, 60, 86, 76, 14, 36, 6, 79, 51, 22 ,35, 43]
of_int2 = of_int.copy()
of_int = [i - 1 for i in of_int]
#plt.scatter(of_int2, Leq_all[np.s_[of_int]])
plt.scatter(bad, Leq_all[np.s_[bad2]], marker = '^')
plt.scatter(good, Leq_all[np.s_[good2]], marker= '*', color = 'red')


#plt.scatter(list(range(1,101)), port_mut)
#plt.scatter(list(range(1,101)), ob_mut)
#plt.scatter(list(range(1,101)), )

xhold3 = []
xhold4 = []
yhold3 = []
yhold4 = []

for i in range(1,101):

    if port_mut[i-1] > ob_mut[i-1]:
        xhold3.append(i)
        yhold3.append(i-1)

    elif port_mut[i-1] < ob_mut[i-1]:
        xhold4.append(i)
        yhold4.append(i-1)
print(xhold1)
print(yhold1)
print(xhold2)
print(yhold2)
print(Leq_all)
# print(np.equal(xhold1,xhold3))
# print(np.equal(xhold2,xhold4))
# print(np.equal(yhold1,yhold3))
# print(np.equal(yhold2,yhold4))





#plt.scatter(xhold1, Leq_all[np.s_[yhold3]])
#plt.scatter(xhold2, Leq_all[np.s_[yhold4]])


plt.xlabel('Port Num')
plt.ylabel('Equivalent Inductance in nH')
plt.title('Equivalent Inductance of Ports Used in Best Solution')
plt.grid(which='both')
plt.legend(['Eq Inductance Shorting Nth Port', 'Eq Inductance Shorting of Center Ports', 'Eq Inductance Shorting of Upper Ports'])
plt.show(block = True)


# test = [21, 81, 7, 59]
# test2 = [18, 47, 62, 10]
# test3 = [50,9,82]
#
# test = np.array([i - 1 + num_pwr for i in test])
# test2 = np.array([i - 1 + num_pwr for i in test2])
# test3 = np.array([i - 1 + num_pwr for i in test3])
#
# print(np.shape(L))
# print(L[0,np.s_[test]])
# print(L[109,np.s_[test]])
# print(L[0,np.s_[test2]])
# print(L[109,np.s_[test2]])
# print(L[0,np.s_[test3]])
# print(L[109,np.s_[test3]])



# z1 = opt.input_net.z
# z2 = opt.input_net2.z
# r_vrm = .003
# l_vrm = 2.5e-9
# vrm_z = r_vrm + 1j*2*opt.freq* math.pi* l_vrm
# vrm_z_array = np.zeros((opt.freq.shape[0],1,1),dtype=complex)
# vrm_z_array[:,0,0] = np.array(vrm_z)
# z2 = pdn1.new_connect(z2, 101, vrm_z_array)
# plt.loglog(opt.freq, np.abs(z1[:,0,0]), '*')
# plt.loglog(opt.freq, np.abs(z2[:,0,0]))
#
# plt.title('Original vs Recreated Impedance',fontsize = 16)
# plt.xlabel('Freq in Hz',fontsize = 16)
# plt.ylabel('Impedance in Ohms', fontsize = 16)
# plt.legend(['Original', 'Recreation'],prop={'size': 14})
# plt.grid(True, which = 'both')
# fig = plt.gcf()
# fig.set_size_inches((12, 9), forward=False)
# plt.show()

