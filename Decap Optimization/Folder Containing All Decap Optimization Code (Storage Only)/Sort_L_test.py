import numpy as np
import matplotlib.pyplot as plt



#L = np.load('100 Port L Mat.npy') # 100 port + 1 actually for the VRM
L = np.load('50 Port L Mat.npz')['L'] # 100 port + 1 actually for the VRM


# Extract out the self L and mutual L of between/of power pins only
# this still also includes all the IC power pins so need to watch out for that
num_IC = 21
num_pwr = 9
num_decap_ports = 51 # includes IC
num_vias = num_IC + num_decap_ports * 2
# delete out ground pins
del_ports = [9 + i for i in range(num_IC - num_pwr)] + [ (num_IC-1) + 2*j for j in range(1,num_decap_ports+1)]

del_ports.reverse()
for i in del_ports:
    L = np.delete(np.delete(L,i,1),i,0)

L = L[8:-1,8:-1]  #For now, take 1 IC pin, and the all the decap power vias. VRM via excluded
self_L = np.diag(L[1:2,1::]) #get self inductance
mutL_toIC = L[0:1, 1::][0]


high_mutual = np.where(mutL_toIC > np.mean(mutL_toIC))[0]
low_mutual = np.where(mutL_toIC < np.mean(mutL_toIC))[0]
print(high_mutual)
print(low_mutual)
sort_m = np.sort(mutL_toIC)
print(np.median(sort_m))


