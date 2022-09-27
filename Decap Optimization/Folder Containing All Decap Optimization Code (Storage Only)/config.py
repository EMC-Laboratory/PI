# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
from pdn_functions import *


class Config(object):
    def __init__(self):

        self.USE_CUDA            = torch.cuda.is_available()
        self.LOAD_SAVED_MOD      = True
        self.LOAD_BEST_MOD       = True
        self.SAVE_TEMP_MODEL     = True
        self.SAVE_BEST_MODEL     = True
        self.MASS_TESTING        = False
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.MODEL               = 'dqn'      # 'PolicyEasyNet' or 'PolicyConvNet'
        self.PROCESS_ID          = 'test_20190514_port_change_freq_change_decaps_lower_z_reward_relative_area_reduction'
        self.SUMMARY_PATH        = "./source/summary/"+self.MODEL+'_'+self.PROCESS_ID

        self.NUM_EPOCHS          = 4000  # was 4000

        self.LOSS_QUEUE_LENGTH   = 10
        self.QUEUE_LENGTH        = 100

        self.PRINT_EVERY         = 1
        self.SAVE_EVERY          = 1

        # for DQN
        self.eps_start           = 1.0
        self.eps_end             = 0
        self.eps_decay_times     = 3000

        # for DQN agent
        self.BUFFER_SIZE = int(2000)        # replay buffer size
        self.BATCH_SIZE = 100               # minibatch size
        self.GAMMA_dqn = 0.4                # discount factor
        self.TAU = 0.1                     # for soft update of target parameters
        self.UPDATE_EVERY = 1               # how often to update the network
        self.LEARNING_RATE       = 0.0000001

        self.best_score_renew_times = 2     # if the score doesn't decrease for how many times, then save the best model

        self.ACTION_NUM          = 10

        self.NUM_WORKERS         = 0
        self.TEST_CASE_NUM       = 1

        self.nf                  = 201
        self.ztarget_file        = np.loadtxt('pdn/ztarget_7_low_freq.txt', skiprows=1)
        self.fstart              = np.min(self.ztarget_file[:, 0]) * 1e6
        self.fstop               = np.max(self.ztarget_file[:, 0]) * 1e6
        self.freq                = np.logspace(np.log10(self.fstart), np.log10(self.fstop), self.nf)
        self.ztarget             = np.interp(self.freq, self.ztarget_file[:, 0]*1e6, self.ztarget_file[:, 1])
        self.Freq                = rf.frequency.Frequency(start=self.fstart/1e6, stop=self.fstop/1e6, npoints=self.nf,
                                                          unit='mhz', sweep_type='log')

        self.input_net = rf.Network('pdn/ASUS-MST_BrdwithCap_mergeIC_port31VRM.s15p').interpolate(self.Freq)
        self.STATE_LEN = self.input_net.z.shape[1] - 1
        self.ic_port = 1
        self.MOST_BEAR_STEP = self.input_net.z.shape[1] - 1  # most bear step is equal to port number
        self.port_priority, _ = prioritize_ports_2(self.input_net, list(range(1, self.input_net.s.shape[1] + 1)),
                                                   ic_port=1)




