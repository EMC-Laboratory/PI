# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:46:54 2020

@author: Ling Zhang, zlyn1993@gmail.com
"""

from copy import deepcopy
import numpy as np
from math import sqrt, pi, sin, cos, log, atan
import skrf as rf
import matplotlib.pyplot as plt

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# merge certain ports in z parameters
def merge_ports(z_orig, map2orig_input, merge_port_list, kept_port=0):
    # merge_port_list, kept_port index begins with 0
    orig_port_list = list(range(0, z_orig.shape[1]))
    del_ports = deepcopy(merge_port_list)
    del_ports.remove(kept_port)
    left_port_list = deepcopy(orig_port_list)
    for a in del_ports:
        left_port_list.remove(a)
    map2orig_output = [map2orig_input[i] for i in left_port_list]
    # calculate inverse z matrix
    z_inv = np.linalg.inv(z_orig)
    if len(orig_port_list) > len(merge_port_list):
        # merge ports by adding the corresponding rows and columns        
        reduce_list = list([kept_port,merge_port_list[-1]+1]) + list(range(merge_port_list[-1]+2,orig_port_list[-1]+1))
        z_inv_merge = np.add.reduceat(np.add.reduceat(z_inv, reduce_list,axis=1),reduce_list,axis=2)
    else:
        z_inv_merge = np.add.reduceat(np.add.reduceat(z_inv, [0],axis=1),[0],axis=2)
    z_merge = np.linalg.inv(z_inv_merge)
    # port_map_orig is the port map to the original port number
    return z_merge, map2orig_output

# short 1 port of S-parameter
def short_1port(input_net, map2orig_input=[0, 1], shorted_port=1):
    # default shorted port for decap is port 1. if input_net is a network, need to point out shorted port #
    short_net = deepcopy(input_net.s11)
    short_net.s = -1*np.ones(short_net.f.shape[0])
    output_net = rf.network.connect(input_net, shorted_port, short_net, 0)
    map2orig_output = deepcopy(map2orig_input)
    del map2orig_output[shorted_port]
    return output_net, map2orig_output

# short several ports of S-parameter
def short_nport(input_net, map2orig_input, shorted_ports):
    output_net = deepcopy(input_net)
    map2orig_output = list(range(0, output_net.s.shape[1]))
    for a in shorted_ports:
        output_net, map2orig_output = short_1port(output_net, map2orig_output,
                                                  shorted_port=map2orig_output.index(a))
    map2orig_output = [map2orig_input[i] for i in map2orig_output]
    return output_net, map2orig_output

def connect_1decap(input_net_z, map2orig_input, connect_port, decap_z11):
    Zaa = deepcopy(input_net_z)
    Zaa = np.delete(Zaa, connect_port, 1)
    Zaa = np.delete(Zaa, connect_port, 2)
    Zpp = input_net_z[:, connect_port, connect_port]
    Zpp = Zpp.reshape((Zpp.shape[0], 1, 1))
    Zqq = decap_z11
    Zap = input_net_z[:, :, connect_port]
    Zap = Zap.reshape((Zap.shape[0], Zap.shape[1], 1))
    Zap = np.delete(Zap, connect_port, 1)
    Zpa = input_net_z[:, connect_port, :]
    Zpa = Zpa.reshape((Zpa.shape[0], 1, Zpa.shape[1]))
    Zpa = np.delete(Zpa, connect_port, 2)
    inv = np.linalg.inv(Zpp+Zqq)
    second = np.einsum('rmn,rkk->rmn', Zap, inv)
    second = np.einsum('rmn,rnd->rmd', second, Zpa)
    output_net_z = Zaa - second
    map2orig_output = deepcopy(map2orig_input)
    del map2orig_output[connect_port]
    return output_net_z, map2orig_output

def connect_z(z1, a_ports, p_ports, z2, q_ports):
    Zaa = z1[np.ix_(list(range(0,z1.shape[0])), a_ports, a_ports)]
    Zpp = z1[np.ix_(list(range(0,z1.shape[0])), p_ports, p_ports)]
    Zap = z1[np.ix_(list(range(0,z1.shape[0])), a_ports, p_ports)]
    Zpa = z1[np.ix_(list(range(0,z1.shape[0])), p_ports, a_ports)]
    Zqq = z2[np.ix_(list(range(0,z1.shape[0])), q_ports, q_ports)]
    z_connect = Zaa - np.einsum('rmn,rnd->rmd', np.einsum('rmn,rnd->rmd', Zap, np.linalg.inv(Zpp+Zqq)), Zpa)
    return z_connect

class PDN():
    def __init__(self):
        self.stackup = np.array([])
        self.die_t = np.array([])
        self.er = 4.4
        
        self.seg_len = np.array([])             # segment length of the boundary
        
        self.outer_bd_node = np.array([])       # N1*2 matrix, rotate counter-clockwise !!!
        self.inner_bd_node = np.array([])       # N2*2 matrix, rotate clockwise !!!
        self.outer_sxy = np.array([])           # S1*4 matrix, rotate counter-clockwise !!!
        self.inner_sxy = np.array([])           # S2*4 matrix, rotate clockwise !!!
        self.sxy = np.array([])                 # S*4 matrix
        self.area = np.array([])                # plane area
        
        self.via_r = 0.1e-3
        
        self.ic_via_xy = np.array([])           # original x,y locations of vias, 2-column matrix
        self.ic_via_type = np.array([])         # original type of vias. 1 for pwr, 0 for gnd
        
        self.decap_via_xy = np.array([])        # x,y locations for decap vias, 2-column matrix
        self.decap_via_type = np.array([])      # type of decap vias, 1 for pwr, 0 for gnd
        self.decap_via_loc = np.array([])       # locations for corresponding decaps, 1 for top layer, 0 for bottom layer
        
        self.via_xy = np.array([])              # x,y locations of all vias
        self.via_type = np.array([])            # type of all vias. 1 for pwr, 0 for gnd
        self.via_loc = np.array([])             # location of all vias. 1 for top, 0 for bot
        
        self.decap = []               # matrix of 6 columns: xp, yp, xg, yg, model_num, top_bot ('top' or 'bot')
        
        self.D = np.array([])                   # (S+via_num)*(S+via_num)
        self.Gh = np.array([])                  # (S+via_num)*(S+via_num)
        self.L_pul = np.array([])               # via_num*via_num
        
        self.C_pul = []
        
        self.R = 0.3                # the worst estimation of distance between two segments
        
        self.z_orig = np.array([])              # original z parameter, without merging IC ports, without adding decaps
        self.z_mergeIC_no_decap = np.array([])           # z parameter with merging IC ports, without adding decaps
        self.z_mergeIC_with_decap = np.array([])                   # z parameter with merging IC ports and added decaps, port #1 is always IC
        
        self.fstart = 0.01e6
        self.fstop = 1e9
        self.nf = 201
        self.freq = rf.frequency.Frequency(start=self.fstart/1e6, stop=self.fstop/1e6, npoints=self.nf,
                                                          unit='mhz', sweep_type='log')
        self.decap_z = np.array([])         # the z matrix forms by all added decaps
        self.decap_list = self.init_decap_library()
        
    def init_decap_library(self):
        decap_list = []
        decap_0, _ = short_1port(rf.Network('readDRLcaps/0_GRM033C80J104KE84.s2p').interpolate(self.freq))
        decap_list.append(decap_0.z)
        decap_1, _ = short_1port(rf.Network('readDRLcaps/1_GRM033R60J474KE90.s2p').interpolate(self.freq))
        decap_list.append(decap_1.z)
        decap_2, _ = short_1port(rf.Network('readDRLcaps/2_GRM155B31C105KA12.s2p').interpolate(self.freq))
        decap_list.append(decap_2.z)
        decap_3, _ = short_1port(rf.Network('readDRLcaps/3_GRM155C70J225KE11.s2p').interpolate(self.freq))
        decap_list.append(decap_3.z)
        decap_4, _ = short_1port(rf.Network('readDRLcaps/4_GRM185C81A475KE11.s2p').interpolate(self.freq))
        decap_list.append(decap_4.z)
        decap_5, _ = short_1port(rf.Network('readDRLcaps/5_GRM188R61A106KAAL.s2p').interpolate(self.freq))
        decap_list.append(decap_5.z)
        decap_6, _ = short_1port(rf.Network('readDRLcaps/6_GRM188B30J226MEA0.s2p').interpolate(self.freq))
        decap_list.append(decap_6.z)
        decap_7, _ = short_1port(rf.Network('readDRLcaps/7_GRM219D80E476ME44.s2p').interpolate(self.freq))
        decap_list.append(decap_7.z)
        decap_8, _ = short_1port(rf.Network('readDRLcaps/8_GRM31CR60J227ME11.s2p').interpolate(self.freq))
        decap_list.append(decap_8.z)
        decap_9, _ = short_1port(rf.Network('readDRLcaps/9_GRM32EC80E337ME05.s2p').interpolate(self.freq))
        decap_list.append(decap_9.z)
        return decap_list
        
    def init_para(self):
        
        self.via_xy = deepcopy(self.ic_via_xy)
        self.via_type = deepcopy(self.ic_via_type)
        self.via_loc = np.ones(self.ic_via_type.shape)
        
    def seg_bd_node(self, bxy, dl):
    # For outer boundary, it must rotate counter-clockwise !!!
    # note that input matrix bxy must go back to the origin point!!!
    # bxy is the boundary coordinate information,N*2 matrix
    # dl is the length of the segments
    # sxy is a S*4 matrix, x1, y1, x2, y2
    
    # if dl is smaller than the interval of the bxy points, do linear
    # interpolation

        bxy_old = deepcopy(bxy)
        if bxy_old[-1,0] != bxy_old[0,0] or bxy_old[-1,1] != bxy_old[0,1]:
            bxy = np.zeros((bxy_old.shape[0]+1,bxy_old.shape[1]))
            bxy[0:-1,:] = bxy_old
            bxy[-1,:] = bxy_old[0,:]
        # calculate number of segments needed first
        nseg = 0
        for i in range(0,bxy.shape[0]-1):
            len_ith = sqrt((bxy[i+1,0]-bxy[i,0])**2+(bxy[i+1,1]-bxy[i,1])**2)
            if dl < len_ith:
                ne = np.floor(len_ith/dl)
                if (len_ith-ne*dl) > dl*0.01:
                    nseg += ne+1
                else:
                    nseg += ne
            else:
                nseg += np.floor(len_ith/dl)
        nseg = nseg.astype(int)
        sxy = np.ndarray((nseg,4))
        s = 0
        for i in range(0,bxy.shape[0]-1):
            len_ith = sqrt((bxy[i+1,0]-bxy[i,0])**2+(bxy[i+1,1]-bxy[i,1])**2)
            if dl < len_ith:
                ne = np.floor(len_ith/dl).astype(int)
                for j in range(0,ne):
                    sxy[s,0] = bxy[i,0] + j*dl/len_ith*(bxy[i+1,0]-bxy[i,0])
                    sxy[s,1] = bxy[i,1] + j*dl/len_ith*(bxy[i+1,1]-bxy[i,1])
                    sxy[s,2] = bxy[i,0] + (j+1)*dl/len_ith*(bxy[i+1,0]-bxy[i,0])
                    sxy[s,3] = bxy[i,1] + (j+1)*dl/len_ith*(bxy[i+1,1]-bxy[i,1])
                    s += 1
                if (len_ith-ne*dl) > dl*0.01:
                    sxy[s,0] = bxy[i,0] + (j+1)*dl/len_ith*(bxy[i+1,0]-bxy[i,0])
                    sxy[s,1] = bxy[i,1] + (j+1)*dl/len_ith*(bxy[i+1,1]-bxy[i,1])
                    sxy[s,2] = bxy[i+1,0]
                    sxy[s,3] = bxy[i+1,1]
                    s += 1
            else:
                sxy[s,0] = bxy[i,0]
                sxy[s,1] = bxy[i,1]
                sxy[s,2] = bxy[i+1,0]
                sxy[s,3] = bxy[i+1,1]
                s += 1
        return sxy
    
    def seg_port(self,x0,y0,r,n=6):
    # Define a function for port segmentation
    # for port boundary, it must rotate clockwise
    # x0, y0 is the port center location, r is radius
    # n is the number of the segments
    # sxy is a S*4 matrix, x1, y1, x2, y2
        dtheta = 2*pi/n
        n = int(n)
        sxy = np.ndarray((n,4))
        for i in range(0,n):
            sxy[i,0] = x0 + r*cos(-(i)*dtheta)
            sxy[i,1] = y0 + r*sin(-(i)*dtheta)
            sxy[i,2] = x0 + r*cos(-(i+1)*dtheta)
            sxy[i,3] = y0 + r*sin(-(i+1)*dtheta)
        return sxy
    
    def seg_bd(self):       
    # segment the outer and inner boundary
    # only support one inner boundary (one void)
        e = 8.85e-12
        self.outer_sxy = self.seg_bd_node(self.outer_bd_node,self.seg_len)
        if self.inner_bd_node.size == 0 and self.inner_sxy.size == 0:
            self.sxy = self.outer_sxy
            self.area = np.array([PolyArea(self.sxy[:,0],self.sxy[:,1])])
        elif self.inner_bd_node.size == 0:
            self.sxy = np.concatenate((self.outer_sxy,self.inner_sxy))
            self.area = np.array([PolyArea(self.outer_sxy[:,0],self.outer_sxy[:,1]) 
            - PolyArea(self.inner_sxy[:,0],self.inner_sxy[:,1])])
        else:
            self.inner_sxy = self.seg_bd_node(self.inner_bd_node,self.seg_len)
            self.sxy = np.concatenate((self.outer_sxy,self.inner_sxy))
            self.area = np.array([PolyArea(self.outer_sxy[:,0],self.outer_sxy[:,1]) 
            - PolyArea(self.inner_sxy[:,0],self.inner_sxy[:,1])])
        self.C_pul = self.er*e*self.area/1
        
    def calc_mat_wo_decap(self):
    # calculate D, L, Gh matrix using for loop, which is time-consuming
    # Especially for many vias, this process is time-consuming
        u = 4*pi*1e-7
        d = 1
        
        Ntot = self.ic_via_xy.shape[0] + self.sxy.shape[0] 
        self.D = np.zeros((Ntot,Ntot))
        self.Gh = np.zeros((Ntot,Ntot))
        self.L_pul = np.zeros((self.ic_via_xy.shape[0],self.ic_via_xy.shape[0]))
        for k in range(0,Ntot):
            for m in range(0,Ntot):
                if m >= self.sxy.shape[0]  and k!=m:
                    self.D[k,m] = 0
                elif m >= self.sxy.shape[0] and k==m:
                    self.D[k,m] = -1
                elif m < self.sxy.shape[0] and k >= self.sxy.shape[0]:
                    xk = self.ic_via_xy[k-self.sxy.shape[0],0]
                    yk = self.ic_via_xy[k-self.sxy.shape[0],1]
                    xa = self.sxy[m,0]
                    ya = self.sxy[m,1]
                    xe = self.sxy[m,2]
                    ye = self.sxy[m,3]
                    lm = sqrt((xa-xe)**2+(ya-ye)**2)
                    v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                    v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                    ra = sqrt((xa-xk)**2+(ya-yk)**2)
                    if np.abs(v2)<1e-10:
                        self.D[k,m] = 0
                    else:
                        self.D[k,m] = v2/pi*(1/sqrt(ra**2-v1**2)*(atan((-v1+lm)/sqrt(ra**2-v1**2))+
                         atan(v1/sqrt(ra**2-v1**2)))-lm/(self.R**2))
                elif m < self.sxy.shape[0] and k < self.sxy.shape[0]:
                    xk = (self.sxy[k,0]+self.sxy[k,2])/2
                    yk = (self.sxy[k,1]+self.sxy[k,3])/2
                    xa = self.sxy[m,0]
                    ya = self.sxy[m,1]
                    xe = self.sxy[m,2]
                    ye = self.sxy[m,3]
                    lm = sqrt((xa-xe)**2+(ya-ye)**2)
                    v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                    v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                    ra = sqrt((xa-xk)**2+(ya-yk)**2)
                    if np.abs(v2)<1e-10:
                        self.D[k,m] = 0
                    else:
                        self.D[k,m] = v2/pi*(1/sqrt(ra**2-v1**2)*(atan((-v1+lm)/sqrt(ra**2-v1**2))+
                         atan(v1/sqrt(ra**2-v1**2)))-lm/(self.R**2))
        for k in range(0,Ntot):
            for m in range(0,Ntot):
                if m >= self.sxy.shape[0] and k==m:
                    self.Gh[k,m] = -u*d/(4*self.area)*(self.via_r**2)*(4*log(self.via_r/self.R)-1)
                elif m >= self.sxy.shape[0] and k!=m:
                    self.Gh[k,m] = 0
                elif m < self.sxy.shape[0] and k >= self.sxy.shape[0]:
                    xk = self.ic_via_xy[k-self.sxy.shape[0],0]
                    yk = self.ic_via_xy[k-self.sxy.shape[0],1]
                    xa = self.sxy[m,0]
                    ya = self.sxy[m,1]
                    xe = self.sxy[m,2]
                    ye = self.sxy[m,3]
                    lm = sqrt((xa-xe)**2+(ya-ye)**2)
                    v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                    v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                    ra = sqrt((xa-xk)**2+(ya-yk)**2)
                    if np.abs(v2)<1e-10:
                        self.Gh[k,m] = 0
                    else:
                        self.Gh[k,m] = u*d*v2/(4*pi*self.area)*(lm*log((lm**2+ra**2-2*v1*lm)/self.R**2)-
                             v1*log((lm**2+ra**2-2*v1*lm)/ra**2)
                             - (1/3*lm**3+ra**2*lm-lm**2*v1)/(2*self.R**2) - 3*lm
                             + 2*sqrt(ra**2-v1**2)*
                             (atan((-v1+lm)/sqrt(ra**2-v1**2))+atan(v1/sqrt(ra**2-v1**2))))
                elif m < self.sxy.shape[0] and k < self.sxy.shape[0]:
                    xk = (self.sxy[k,0]+self.sxy[k,2])/2
                    yk = (self.sxy[k,1]+self.sxy[k,3])/2
                    xa = self.sxy[m,0]
                    ya = self.sxy[m,1]
                    xe = self.sxy[m,2]
                    ye = self.sxy[m,3]
                    lm = sqrt((xa-xe)**2+(ya-ye)**2)
                    v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                    v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                    ra = sqrt((xa-xk)**2+(ya-yk)**2)
                    if np.abs(v2)<1e-10:
                        self.Gh[k,m] = 0
                    else:
                        self.Gh[k,m] = u*d*v2/(4*pi*self.area)*(lm*log((lm**2+ra**2-2*v1*lm)/self.R**2)-
                             v1*log((lm**2+ra**2-2*v1*lm)/ra**2)
                             - (1/3*lm**3+ra**2*lm-lm**2*v1)/(2*self.R**2) - 3*lm
                             + 2*sqrt(ra**2-v1**2)*
                             (atan((-v1+lm)/sqrt(ra**2-v1**2))+atan(v1/sqrt(ra**2-v1**2))))
        E_D_inv = np.linalg.inv(np.identity(Ntot)-self.D)
        Gh_k = np.sum(self.Gh, axis=1)
        G_k = np.zeros((Ntot))
        # Excite each via and obtain inductance
        for j in range(0,self.ic_via_xy.shape[0]):
            m = j + self.sxy.shape[0]
            for k in range(0,Ntot):
                if k==m:
                    G_k[k] = -u*d/pi*log(self.via_r/self.R)
                elif k!=m and k>=self.sxy.shape[0]:
                    r_km = sqrt((self.ic_via_xy[m-self.sxy.shape[0],0]-self.ic_via_xy[k-self.sxy.shape[0],0])**2
                                +(self.ic_via_xy[m-self.sxy.shape[0],1]-self.ic_via_xy[k-self.sxy.shape[0],1])**2)
                    G_k[k] = -u*d/pi*(log(r_km/self.R)-r_km**2/(2*self.R**2))
                else:
                    xm = self.ic_via_xy[m-self.sxy.shape[0],0]
                    ym = self.ic_via_xy[m-self.sxy.shape[0],1]
                    xk = (self.sxy[k,0]+self.sxy[k,2])/2
                    yk = (self.sxy[k,1]+self.sxy[k,3])/2
                    r_km = sqrt((xm-xk)**2+(ym-yk)**2)
                    G_k[k] = -u*d/pi*(log(r_km/self.R)-r_km**2/(2*self.R**2))
            G = Gh_k + G_k
            L = np.dot(E_D_inv,G)
            self.L_pul[:,j] = L[self.sxy.shape[0]:Ntot]
        # add internal inductance
        for i in range(0,self.L_pul.shape[0]):
            self.L_pul[i,i] += u*d/(8*pi)
        self.z_orig = self.calc_z_from_Lpul(self.stackup, self.via_type, self.via_loc, 
                                  self.die_t, self.L_pul, self.C_pul, self.freq.f)
        if np.where(self.ic_via_type==1)[0].shape[0] > 1:
            self.z_mergeIC_no_decap,_ = merge_ports(self.z_orig, list(range(0,np.where(self.ic_via_type==1)[0].shape[0])), 
                                   list(range(0,np.where(self.ic_via_type==1)[0].shape[0])))
        else:
            self.z_mergeIC_no_decap = self.z_orig
    
    def add_via(self, via_x, via_y, via_type, via_loc):
        d = 1
        u = 4*pi*1e-7
        via_xy_new = np.ndarray((self.via_xy.shape[0]+1,2))
        via_xy_new[0:self.via_xy.shape[0],:] = self.via_xy
        via_xy_new[self.via_xy.shape[0],0] = via_x
        via_xy_new[self.via_xy.shape[0],1] = via_y
        self.via_xy = via_xy_new
        via_type_new = np.ndarray((self.via_type.shape[0]+1))
        via_type_new[0:self.via_type.shape[0]] = self.via_type
        via_type_new[self.via_type.shape[0]] = via_type
        self.via_type = via_type_new
        via_loc_new = np.ndarray((self.via_loc.shape[0]+1))
        via_loc_new[0:self.via_loc.shape[0]] = self.via_loc
        via_loc_new[self.via_loc.shape[0]] = via_loc
        self.via_loc = via_loc_new
        
        D_new = np.zeros([self.D.shape[0]+1,self.D.shape[0]+1])
        Gh_new = np.zeros([self.Gh.shape[0]+1,self.Gh.shape[0]+1])
        L_pul_new = np.zeros([self.L_pul.shape[0]+1,self.L_pul.shape[0]+1])
        G_k_new = np.zeros((self.D.shape[0]+1))
        D_new[0:self.D.shape[0],0:self.D.shape[0]] = self.D
        Gh_new[0:self.Gh.shape[0],0:self.Gh.shape[0]] = self.Gh
        L_pul_new[0:self.L_pul.shape[0],0:self.L_pul.shape[0]] = self.L_pul
        D_new[0:self.D.shape[0],self.D.shape[0]] = 0
        D_new[self.D.shape[0],self.D.shape[0]] = -1
        Gh_new[0:self.D.shape[0],self.D.shape[0]] = 0
        Gh_new[self.D.shape[0],self.D.shape[0]] = -u*d/(4*
              self.area)*(self.via_r**2)*(4*log(self.via_r/self.R)-1)
        
        for m in range(0,self.D.shape[0]):
            k = self.D.shape[0]
            if m >= self.sxy.shape[0]:
                D_new[k,m] = 0
                Gh_new[k,m] = 0
            else:
                xk = self.via_xy[k-self.sxy.shape[0],0]
                yk = self.via_xy[k-self.sxy.shape[0],1]
                xa = self.sxy[m,0]
                ya = self.sxy[m,1]
                xe = self.sxy[m,2]
                ye = self.sxy[m,3]
                lm = sqrt((xa-xe)**2+(ya-ye)**2)
                v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                ra = sqrt((xa-xk)**2+(ya-yk)**2)
                if np.abs(v2)<1e-10:
                    D_new[k,m] = 0
                    Gh_new[k,m] = 0
                else:
                    D_new[k,m] = v2/pi*(1/sqrt(ra**2-v1**2)*(atan((-v1+lm)/sqrt(ra**2-v1**2))+
                     atan(v1/sqrt(ra**2-v1**2)))-lm/(self.R**2))
                    Gh_new[k,m] = u*d*v2/(4*pi*self.area)*(lm*log((lm**2+ra**2-2*v1*lm)/self.R**2)-
                             v1*log((lm**2+ra**2-2*v1*lm)/ra**2)
                             - (1/3*lm**3+ra**2*lm-lm**2*v1)/(2*self.R**2) - 3*lm
                             + 2*sqrt(ra**2-v1**2)*
                             (atan((-v1+lm)/sqrt(ra**2-v1**2))+atan(v1/sqrt(ra**2-v1**2))))
        self.D = D_new
        self.Gh = Gh_new
        E_D_inv = np.linalg.inv(np.identity(self.D.shape[0])-self.D)
        Gh_k_new = np.sum(self.Gh, axis=1)
        for k in range(0,G_k_new.shape[0]):
            m = G_k_new.shape[0]-1
            if k==m:
                G_k_new[k] = -u*d/pi*log(self.via_r/self.R)
            elif k!=m and k>=self.sxy.shape[0]:
                r_km = sqrt((self.via_xy[m-self.sxy.shape[0],0]-self.via_xy[k-self.sxy.shape[0],0])**2
                            +(self.via_xy[m-self.sxy.shape[0],1]-self.via_xy[k-self.sxy.shape[0],1])**2)
                G_k_new[k] = -u*d/pi*(log(r_km/self.R)-r_km**2/(2*self.R**2))
            else:
                xm = self.via_xy[m-self.sxy.shape[0],0]
                ym = self.via_xy[m-self.sxy.shape[0],1]
                xk = (self.sxy[k,0]+self.sxy[k,2])/2
                yk = (self.sxy[k,1]+self.sxy[k,3])/2
                r_km = sqrt((xm-xk)**2+(ym-yk)**2)
                G_k_new[k] = -u*d/pi*(log(r_km/self.R)-r_km**2/(2*self.R**2))
        G = Gh_k_new + G_k_new
        L = np.dot(E_D_inv,G)
        L_pul_new[:,G_k_new.shape[0]-self.sxy.shape[0]-1] = L[self.sxy.shape[0]:G_k_new.shape[0]]
        L_pul_new[G_k_new.shape[0]-self.sxy.shape[0]-1,:] = L[self.sxy.shape[0]:G_k_new.shape[0]]
        self.L_pul = L_pul_new
        # add self inductance
        self.L_pul[self.L_pul.shape[0]-1,self.L_pul.shape[0]-1] += u*d/(8*pi)
        
    def calc_z_from_Lpul(self, stackup, via_type, via_loc, die_t, L_pul, C_pul, freq):
        # only support the top layer and bottom layer are gnd for now !!!!
        # find the node index of the first gnd via and use it as the reference node
        # via_loc is the location for the via, whether on the top(1) or bottom(0) layer
        # Actually only the via_loc for power vias will matter
        # IC vias must be on the top layer
        branch_num = die_t.shape[0]*(via_type.shape[0]+1)
        node_mat = np.ndarray((stackup.shape[0],via_type.shape[0]+1),dtype=int)
        top_port_node_list = []
        bot_port_node_list = []
        node = 0
        for r in range(0,stackup.shape[0]):
            for c in range(0,via_type.shape[0]):
                if c == 0:
                    first_same_node = 1
                if stackup[r] != via_type[c]:
                    node_mat[r,c] = node
                    node += 1
                elif stackup[r] == via_type[c] and first_same_node == 1:
                    node_mat[r,c] = node
                    same_node_num = node
                    first_same_node = 0
                    node += 1
                elif stackup[r] == via_type[c] and first_same_node == 0:
                    node_mat[r,c] = same_node_num
                if r==0 and c==via_type.shape[0]-1:
                    top_ref_node = same_node_num
                if r==stackup.shape[0]-1 and c==via_type.shape[0]-1:
                    bot_ref_node = same_node_num
                if r==0 and via_type[c]==1 and via_loc[c]==1:
                    top_port_node_list.append(node_mat[r,c])
                if r==stackup.shape[0]-1 and via_type[c]==1 and via_loc[c]==0:
                    bot_port_node_list.append(node_mat[r,c])
                    
            c = via_type.shape[0]
            node_mat[r,c] = same_node_num
                    
        node_num = node
        A = np.zeros((freq.shape[0],node_num,branch_num))    # reduced incidence matrix
        for c in range(0,branch_num):
            r_num = np.floor(c/(via_type.shape[0]+1)).astype(int)
            c_num = np.remainder(c,via_type.shape[0]+1)
            begin_node = node_mat[r_num,c_num]
            end_node = node_mat[r_num+1,c_num]
            A[:,begin_node,c] = 1
            A[:,end_node,c] = -1
            
        Zb = np.zeros((freq.shape[0],branch_num,branch_num),dtype=complex)
        
        for i in range(0,die_t.shape[0]):
            Zb[:,i*(via_type.shape[0]+1):(i+1)*(via_type.shape[0]+1)-1,
               i*(via_type.shape[0]+1):(i+1)*(via_type.shape[0]+1)-1] = 1j*2*pi*die_t[i]*np.einsum('i,jk->ijk', 
               freq, L_pul)
            Zb[:,(i+1)*(via_type.shape[0]+1)-1,(i+1)*(via_type.shape[0]+1)-1] = 1/(1j*2*pi*freq*C_pul)*die_t[i]
        
        Yb = np.linalg.inv(Zb)
        Yn = np.einsum('rmn,rnk->rmk', np.einsum('rmn,rnk->rmk', A, Yb), np.transpose(A,(0,2,1)))
        
        Yn_reduce = np.delete(np.delete(Yn,[top_ref_node],axis=2),[top_ref_node],axis=1)
        ztot = np.linalg.inv(Yn_reduce)
        
        top_port_node_list = [x for x in top_port_node_list if x<top_ref_node] + [x-1 for x in top_port_node_list if x>top_ref_node]
        bot_port_node_list = [x-1 for x in bot_port_node_list]
        bot_ref_node -= 1
        
        # subtract the corresponding rows and columns at the bottom reference node, for bottom decap vias
        for x in bot_port_node_list:
            x = int(x)
            ztot[:,x,:] = ztot[:,x,:] - ztot[:,bot_ref_node,:]
            ztot[:,:,x] = ztot[:,:,x] - ztot[:,:,bot_ref_node]
            
        top_node = 0
        bot_node = 0
        power_port_list = []
        for x in range(0,via_type.shape[0]):
            if via_type[x] == 1 and via_loc[x] == 1:
                power_port_list.append(top_port_node_list[top_node])
                top_node += 1
            elif via_type[x] == 1 and via_loc[x] == 0:
                power_port_list.append(bot_port_node_list[bot_node])
                bot_node += 1

        z = ztot[np.ix_(list(range(0,ztot.shape[0])), 
                        power_port_list, power_port_list)]
        return z
    
    def add_decap(self, xp, yp, xg, yg, model_num, top_bot):
        self.add_via(xp,yp,1,top_bot)
        self.add_via(xg,yg,0,top_bot)
        self.decap.append(np.array([xp, yp, xg, yg, model_num, top_bot]))
        if len(self.decap) == 1:
            self.decap_z = np.ndarray((self.freq.f.shape[0],1,1))
            self.decap_z = self.decap_list[model_num]
        else:
            decap_z_new = np.zeros((self.freq.f.shape[0],len(self.decap),len(self.decap)),dtype=complex)
            decap_z_new[:,0:len(self.decap)-1,0:len(self.decap)-1] = self.decap_z
            decap_z_new[:,len(self.decap)-1,len(self.decap)-1] = np.squeeze(self.decap_list[model_num])
            self.decap_z = decap_z_new
        self.z_orig = self.calc_z_from_Lpul(self.stackup, self.via_type, self.via_loc, 
                                  self.die_t, self.L_pul, self.C_pul, self.freq.f)
        if np.where(self.ic_via_type==1)[0].shape[0] > 1: # if more than 1 IC power vias
            self.z_mergeIC_no_decap, _ = merge_ports(self.z_orig, list(range(0,self.z_orig.shape[1])), 
                                   list(range(0,np.where(self.ic_via_type==1)[0].shape[0])))
        else:
            self.z_mergeIC_no_decap = deepcopy(self.z_orig)
        self.z_mergeIC_with_decap = connect_z(self.z_mergeIC_no_decap, [0], list(range(1,1+len(self.decap))), 
                           self.decap_z, list(range(0,len(self.decap))))
        
    def calc_z(self):
        e = 8.85e-12
        if self.area.shape[0] == 0:
            self.area = np.array([PolyArea(self.sxy[:,0],self.sxy[:,1])])
            self.C_pul = np.array([self.er*e*self.area/1])
        # directly calculate z parameters given decap via locations
        via_xy = np.concatenate((self.ic_via_xy,self.decap_via_xy),axis=0)
        via_type = np.concatenate((self.ic_via_type,self.decap_via_type),axis=0)
        via_loc = np.concatenate((np.ones(self.ic_via_type.shape),self.decap_via_loc),axis=0)
        
        u = 4*pi*1e-7
        d = 1
        
        Ntot = via_xy.shape[0] + self.sxy.shape[0] 
        self.D = np.zeros((Ntot,Ntot))
        self.Gh = np.zeros((Ntot,Ntot))
        self.L_pul = np.zeros((via_xy.shape[0],via_xy.shape[0]))
        for k in range(0,Ntot):
            for m in range(0,Ntot):
                if m >= self.sxy.shape[0]  and k!=m:
                    self.D[k,m] = 0
                elif m >= self.sxy.shape[0] and k==m:
                    self.D[k,m] = -1
                elif m < self.sxy.shape[0] and k >= self.sxy.shape[0]:
                    xk = via_xy[k-self.sxy.shape[0],0]
                    yk = via_xy[k-self.sxy.shape[0],1]
                    xa = self.sxy[m,0]
                    ya = self.sxy[m,1]
                    xe = self.sxy[m,2]
                    ye = self.sxy[m,3]
                    lm = sqrt((xa-xe)**2+(ya-ye)**2)
                    v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                    v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                    ra = sqrt((xa-xk)**2+(ya-yk)**2)
                    #if np.abs(v2)<1e-10:
                    if np.abs(v1**2-ra**2) < 1e-10:
                        self.D[k,m] = 0
                    else:
                        self.D[k,m] = v2/pi*(1/sqrt(ra**2-v1**2)*(atan((-v1+lm)/sqrt(ra**2-v1**2))+
                         atan(v1/sqrt(ra**2-v1**2)))-lm/(self.R**2))
                elif m < self.sxy.shape[0] and k < self.sxy.shape[0]:
                    xk = (self.sxy[k,0]+self.sxy[k,2])/2
                    yk = (self.sxy[k,1]+self.sxy[k,3])/2
                    xa = self.sxy[m,0]
                    ya = self.sxy[m,1]
                    xe = self.sxy[m,2]
                    ye = self.sxy[m,3]
                    
                    #pdb.set_trace()
                    
                    lm = sqrt((xa-xe)**2+(ya-ye)**2)
                    
                    v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                    v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                    ra = sqrt((xa-xk)**2+(ya-yk)**2)
                    #if np.abs(v2)<1e-10:
                    if np.abs(v1**2-ra**2) < 1e-10:
                        self.D[k,m] = 0
                    else:
                        self.D[k,m] = v2/pi*(1/sqrt(ra**2-v1**2)*(atan((-v1+lm)/sqrt(ra**2-v1**2))+
                         atan(v1/sqrt(ra**2-v1**2)))-lm/(self.R**2))
        for k in range(0,Ntot):
            for m in range(0,Ntot):
                if m >= self.sxy.shape[0] and k==m:
                    self.Gh[k,m] = -u*d/(4*self.area)*(self.via_r**2)*(4*log(self.via_r/self.R)-1)
                elif m >= self.sxy.shape[0] and k!=m:
                    self.Gh[k,m] = 0
                elif m < self.sxy.shape[0] and k >= self.sxy.shape[0]:
                    xk = via_xy[k-self.sxy.shape[0],0]
                    yk = via_xy[k-self.sxy.shape[0],1]
                    xa = self.sxy[m,0]
                    ya = self.sxy[m,1]
                    xe = self.sxy[m,2]
                    ye = self.sxy[m,3]
                    lm = sqrt((xa-xe)**2+(ya-ye)**2)
                    v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                    v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                    ra = sqrt((xa-xk)**2+(ya-yk)**2)
                    #if np.abs(v2)<1e-10:
                    if np.abs(v1**2-ra**2) < 1e-10:
                        self.Gh[k,m] = 0
                    else:
                        self.Gh[k,m] = u*d*v2/(4*pi*self.area)*(lm*log((lm**2+ra**2-2*v1*lm)/self.R**2)-
                             v1*log((lm**2+ra**2-2*v1*lm)/ra**2)
                             - (1/3*lm**3+ra**2*lm-lm**2*v1)/(2*self.R**2) - 3*lm
                             + 2*sqrt(ra**2-v1**2)*
                             (atan((-v1+lm)/sqrt(ra**2-v1**2))+atan(v1/sqrt(ra**2-v1**2))))
                elif m < self.sxy.shape[0] and k < self.sxy.shape[0]:
                    xk = (self.sxy[k,0]+self.sxy[k,2])/2
                    yk = (self.sxy[k,1]+self.sxy[k,3])/2
                    xa = self.sxy[m,0]
                    ya = self.sxy[m,1]
                    xe = self.sxy[m,2]
                    ye = self.sxy[m,3]
                    lm = sqrt((xa-xe)**2+(ya-ye)**2)
                    v1 = 1/lm*((xk-xa)*(xe-xa)+(yk-ya)*(ye-ya))
                    v2 = -1/lm*((xk-xa)*(ye-ya)-(yk-ya)*(xe-xa))
                    ra = sqrt((xa-xk)**2+(ya-yk)**2)
                    #if np.abs(v2)<1e-10:
                    if np.abs(v1**2-ra**2) < 1e-10:
                        self.Gh[k,m] = 0
                    else:
                        self.Gh[k,m] = u*d*v2/(4*pi*self.area)*(lm*log((lm**2+ra**2-2*v1*lm)/self.R**2)-
                             v1*log((lm**2+ra**2-2*v1*lm)/ra**2)
                             - (1/3*lm**3+ra**2*lm-lm**2*v1)/(2*self.R**2) - 3*lm
                             + 2*sqrt(ra**2-v1**2)*
                             (atan((-v1+lm)/sqrt(ra**2-v1**2))+atan(v1/sqrt(ra**2-v1**2))))
        E_D_inv = np.linalg.inv(np.identity(Ntot)-self.D)
        Gh_k = np.sum(self.Gh, axis=1)
        G_k = np.zeros((Ntot))
        # Excite each via and obtain inductance
        for j in range(0,via_xy.shape[0]):
            m = j + self.sxy.shape[0]
            for k in range(0,Ntot):
                if k==m:
                    G_k[k] = -u*d/pi*log(self.via_r/self.R)
                elif k!=m and k>=self.sxy.shape[0]:
                    r_km = sqrt((via_xy[m-self.sxy.shape[0],0]-via_xy[k-self.sxy.shape[0],0])**2
                                +(via_xy[m-self.sxy.shape[0],1]-via_xy[k-self.sxy.shape[0],1])**2)
                    G_k[k] = -u*d/pi*(log(r_km/self.R)-r_km**2/(2*self.R**2))
                else:
                    xm = via_xy[m-self.sxy.shape[0],0]
                    ym = via_xy[m-self.sxy.shape[0],1]
                    xk = (self.sxy[k,0]+self.sxy[k,2])/2
                    yk = (self.sxy[k,1]+self.sxy[k,3])/2
                    r_km = sqrt((xm-xk)**2+(ym-yk)**2)
                    G_k[k] = -u*d/pi*(log(r_km/self.R)-r_km**2/(2*self.R**2))
            G = Gh_k + G_k
            L = np.dot(E_D_inv,G)
            self.L_pul[:,j] = L[self.sxy.shape[0]:Ntot]
        # add internal inductance
        for i in range(0,self.L_pul.shape[0]):
            self.L_pul[i,i] += u*d/(8*pi)
            
        self.z_orig = self.calc_z_from_Lpul(self.stackup, via_type, via_loc, self.die_t, 
                                  self.L_pul, self.C_pul, self.freq.f)
        if np.where(self.ic_via_type==1)[0].shape[0] > 1: # if more than 1 IC power vias
            self.z_mergeIC_no_decap, _ = merge_ports(self.z_orig, list(range(0,self.z_orig.shape[1])), 
                                   list(range(0,np.where(self.ic_via_type==1)[0].shape[0])))
        else:
            self.z_mergeIC_no_decap = deepcopy(self.z_orig)
        return self.z_mergeIC_no_decap
        
    def connect_n_decap(self, input_z, map2orig_input, connect_port_list, decap_num_list):
        
        if len(connect_port_list) > 0:
            output_z = deepcopy(input_z)
            map2orig_output = list(range(0, output_z.shape[1]))
            for x in range(0, len(connect_port_list)):
                output_z, map2orig_output = connect_1decap(output_z, map2orig_output, 
                                                           connect_port=map2orig_output.index(connect_port_list[x]), 
                                                           decap_z11=self.decap_list[decap_num_list[x]])
            map2orig_output = [map2orig_input[i] for i in map2orig_output]
        else:
            output_z = deepcopy(input_z)
            map2orig_output = deepcopy(map2orig_input)
        return output_z, map2orig_output

    def save2s(self, z, filename, z0=50):
        brd = rf.Network()
        brd.z0 = z0
        brd.frequency = self.freq
        brd.s = rf.network.z2s(z)
        brd.write_touchstone(filename + ".s" + str(z.shape[1]) + "p")
        
    def plot_z(self, z, port1=0, port2=0):
        plt.loglog(self.freq.f/1e6,np.abs(z[:,port1,port2]))
        plt.grid(which='both')
        plt.xlabel('Frequency(MHz)')
        plt.ylabel('Impedance(Ohm)')
        plt.show()
        
    