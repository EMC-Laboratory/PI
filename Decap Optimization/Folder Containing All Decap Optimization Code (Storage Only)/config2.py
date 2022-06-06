# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import skrf as rf
import numpy as np
import copy


class Config(object):
    def __init__(self):

        self.USE_CUDA            = torch.cuda.is_available()
        self.LOSS_QUEUE_LENGTH   = 10
        self.QUEUE_LENGTH        = 100
        self.PRINT_EVERY         = 1
        self.SAVE_EVERY          = 1
        self.nf                  = 201
        self.nf2                 = 402
        #self.ztarget_file = np.loadtxt('pdn/Arb.txt', skiprows=1)
        #self.ztarget_file        = np.loadtxt('pdn/S27Pztarget.txt', skiprows=1)
        #self.ztarget_file        = np.loadtxt('pdn/42PortTest1.txt', skiprows=1)
        self.ztarget_file       = np.loadtxt('pdn/zedit.txt', skiprows=1)

        self.fstart              = .01 * 1e6
        self.fstop               = 20* 1e6

        self.fstart2 = .01 * 1e6
        self.fstop2 = 20 * 1e6




        # self.fstart = 0.01e6
        # self.fstop = 1e9
        #self.nf = 501

        self.freq                = np.logspace(np.log10(self.fstart), np.log10(self.fstop), self.nf)
        self.freq2 = np.logspace(np.log10(self.fstart2), np.log10(self.fstop2), self.nf2)


        self.ztarget             = np.interp(self.freq, self.ztarget_file[:, 0]*1e6, self.ztarget_file[:, 1])
        #self.ztarget2            = np.interp(self.freq, self.ztarget_file2[:, 0] * 1e6, self.ztarget_file2[:, 1])

        self.Freq                = rf.frequency.Frequency(start=self.fstart/1e6, stop=self.fstop/1e6, npoints=self.nf,
                                                          unit='mhz', sweep_type='log') # Used to set interpolation
        self.Freq2 = rf.frequency.Frequency(start=self.fstart2 / 1e6, stop=self.fstop2 / 1e6, npoints=self.nf2,
                                           unit='mhz', sweep_type='log')  # Used to set interpolation



        #self.input_net = rf.Network('pdn/ASUS-MST_BrdwithCap_mergeIC_port31VRM.s15p').interpolate(self.Freq)  #15 total ports 14 decaps
        #self.input_net = rf.Network('pdn/50_Cap_Test.s51p').interpolate(self.Freq)
        self.input_net = rf.Network('new_data_test_to_compare_methods/75 Caps 1.s76p').interpolate(self.Freq)

        #self.input_net = rf.Network('pdn/50_Cap_Test2 1000.s51p').interpolate(self.Freq)
        #self.input_net = rf.Network('pdn/50_Cap_Test7 1103.s51p').interpolate(self.Freq)

        #self.input_net = rf.Network('pdn/Case0.s21p').interpolate(self.Freq)

        #self.input_net = rf.Network('pdn/10200.s21p').interpolate(self.Freq)
        #self.input_net = rf.Network('pdn/5000.s21p').interpolate(self.Freq)

        #self.input_net = rf.Network('pdn/1.s101p').interpolate(self.Freq)
        #self.input_net = rf.Network('100 Port Board CCW S-Parameters.s101p').interpolate(self.Freq)
        #self.input_net2 = rf.Network('100 Port Merge IC.s102p').interpolate(self.Freq)
        #self.input_net3 = rf.Network('Test for L with VRM 5 GHz.s2p').interpolate(self.Freq)
        #self.input_net = rf.Network('pdn/50_Cap_Test.s51p').interpolate(self.Freq)

        #self.input_net2 = rf.Network('pdn/1.s101p').interpolate(self.Freq)

        #self.input_net = rf.Network('pdn/1.s101p').interpolate(self.Freq)
        self.total_ports = len(self.input_net.z[1])  # total number of ports including observation
                                                     # with network objects.z, the 1st index dimension= # of ports, selects port location
                                                     # this may change if network code changes/updates or we don't use network anymore

        self.decap_ports = self.total_ports - 1  # total number of ports where you can stick a decap in

        self.ic_port = 1                         # port number from where you are observing
                                                 # key point. While IC port is port 1, it corresponds to index 0
                                                 # of an array. So when we put decaps in, we start at index 1, which is techinically port 2

        self.port_nums = [i for i in range(2,self.total_ports)]

        self.connect_order = copy.deepcopy(self.port_nums).reverse()
        # this one isn't used right now
        # but this was for if the observation port wasn't port 1, then when I connect a certain ports
        # the network port number collapses down. Can be avoided if always connecting from right side,
        # and skipping the IC port

        self.fail_score = 1
        self.no_empty_port_score = 10

        self.decaps = ['GRM033C80J104KE84.s2p',
                  'GRM033R60J474KE90.s2p',
                  'GRM155B31C105KA12.s2p',
                  'GRM155C70J225KE11.s2p',
                  'GRM185C81A475KE11.s2p',
                  'GRM188R61A106KAAL.s2p',
                  'GRM188B30J226MEA0.s2p',
                  'GRM219D80E476ME44.s2p',
                  'GRM31CR60J227ME11.s2p',
                  'GRM32EC80E337ME05.s2p']
        self.num_decaps = len(self.decaps)  # 10 decaps in library.
# 200
# 192
# 186
# 168
# 155
# 144
# 132
# 118
# 91
# 86
