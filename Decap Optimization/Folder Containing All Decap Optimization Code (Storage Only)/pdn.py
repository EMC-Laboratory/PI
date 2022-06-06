# coding: utf-8
# Author: Ling Zhang
# Email : lzd76@mst.edu

import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

def find_optimum_min(opt):
    impedance_points = opt.freq
    optimum_min_pt = 0
    print('Minimum z target point is', np.amax(impedance_points))
    print('Maximum z target point it', np.amin(impedance_points))
    if np.amax(opt.ztarget) == np.amin(opt.ztarget):  # constant curve
        optimum_min_pt = np.floor(opt.nf / 2).astype(np.int64)
        print('Minimum z should be around', opt.freq[optimum_min_pt], 'Hz')
    else:  # works for impedance curve changing once ( increasing, 27 port) for now only
        for index in range(opt.nf - 1):
            if opt.ztarget[index + 1] - opt.ztarget[index] != 0:
                optimum_min_pt = index
                print('Minimum z should be around', opt.freq[optimum_min_pt], 'Hz')
                break
    return optimum_min_pt

def short_1port(input_net, shorted_port=2):
    # default shorted port for decap is port 2. if input_net is a network, need to point out shorted port #
    short_net = copy.deepcopy(input_net.s11)
    short_net.s = -1*np.ones(short_net.f.shape[0])
    output_net = rf.network.connect(input_net, shorted_port-1, short_net, 0)

    return output_net


def select_decap(decap_num, opt):
    if decap_num == 0:
        decap = None
    elif decap_num == 1:
        decap = short_1port(rf.Network('decap/decap_s2p_v2/GRM033C80J104KE84.s2p').interpolate(opt.Freq))
    elif decap_num == 2:
        decap= short_1port(rf.Network('decap/decap_s2p_v2/GRM033R60J474KE90.s2p').interpolate(opt.Freq))
    elif decap_num == 3:
        decap= short_1port(rf.Network('decap/decap_s2p_v2/GRM155B31C105KA12.s2p').interpolate(opt.Freq))
    elif decap_num == 4:
        decap = short_1port(rf.Network('decap/decap_s2p_v2/GRM155C70J225KE11.s2p').interpolate(opt.Freq))
    elif decap_num == 5:
        decap = short_1port(rf.Network('decap/decap_s2p_v2/GRM185C81A475KE11.s2p').interpolate(opt.Freq))
    elif decap_num == 6:
        decap = short_1port(rf.Network('decap/decap_s2p_v2/GRM188R61A106KAAL.s2p').interpolate(opt.Freq))
    elif decap_num == 7:
        decap = short_1port(rf.Network('decap/decap_s2p_v2/GRM188B30J226MEA0.s2p').interpolate(opt.Freq))
    elif decap_num == 8:
        decap = short_1port(rf.Network('decap/decap_s2p_v2/GRM219D80E476ME44.s2p').interpolate(opt.Freq))
    elif decap_num == 9:
        decap = short_1port(rf.Network('decap/decap_s2p_v2/GRM31CR60J227ME11.s2p').interpolate(opt.Freq))
    elif decap_num == 10:
        decap = short_1port(rf.Network('decap/decap_s2p_v2/GRM32EC80E337ME05.s2p').interpolate(opt.Freq))
    return decap





def connect_1decap(target_net, decap_sll, port_num):
    #Theoretically how it works is, the first decap will be connected to the first port
    #making a n port network have n-1 ports, with the first port open again on new network
    output_net = copy.deepcopy(target_net)
    if decap_sll is not None: # connect decap if there is a decap to connect
        #print("Num Ports Currently:", output_net.nports, "Port:", port_num, "to be used")
        output_net = rf.network.connect(target_net, port_num, decap_sll, 0)
    return output_net

def connect_n_decap(solution, decode_map, opt):
    target_net = copy.deepcopy(opt.input_net)  # asus board network object to connect to
    port_num = opt.ic_port   # port_num will be used to tell what port you'll start connecting from
    # NOTE TO SELF, 7/7/19, I JUST NOTICED BUT THIS WOULDN'T WORK IF YOU ARE NOT MEASURING FROM PORT 1.
    # HAVE TO BE AWARE OF THAT. CAN BE FIXED BY CONNECTING DECAPS STARTING AT THE LARGEST PORT NUMBER
    # THAT WAY WHEN THE PORT NUMBERS SHIFTS, THE PORT NUMBERS < THE LARGEST PORT NUMBER WON'T CHANGE

    for decap in solution:    # Each gene (decap) of a chromosome solution
        decap_slln = decap    # gives encoded number, 0 - 10, of capacitor to be connected to port number: port_num
        target_net = connect_1decap(target_net, decode_map[decap_slln], port_num)
        # I don't have to make this two lines but makes more sense to me when reading.

        if decap_slln == 0:
            port_num = port_num + 1
            # connecting a decap will shift the ports on the network object inwards.
            # if a port is selected to not have a decap, then the ports layout doesn't shift
            # I have to manually shift what port I connect to next.
    solu_net = target_net  # Gets the fully connected network
    #print(decap_map)
    return solu_net


def new_connect_1decap(input_net_z, connect_port, decap_z11):
    output_net_z = new_connect(input_net_z, connect_port, decap_z11)
    return output_net_z

def new_connect_n_decap(input_net_z, decap_map, caps_objs_z, opt):
    # formally used to connect decaps
    decap_map_z = copy.deepcopy(input_net_z)
    port_num = opt.ic_port   # Has the problem where this only works if IC port is the very first port. Okay for now but need generalize

    for decap in decap_map:
        if decap != 0:
            #decap_map_z = new_connect_1decap(decap_map_z, port_num, caps_objs_z[decap-1])
            decap_map_z = new_connect(decap_map_z, port_num, caps_objs_z[decap-1])
        if decap == 0:
            port_num = port_num + 1
    return decap_map_z

def new_connect(input_net_z, connect_port, decap_z11):
    Zaa = copy.deepcopy(input_net_z)
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

    inv = np.linalg.inv(Zpp + Zqq)

    second = np.einsum('rmn,rkk->rmn', Zap, inv)
    second = np.einsum('rmn,rnd->rmd', second, Zpa)

    output_net_z = Zaa - second

    return output_net_z  # z parameters returned

def calc_z(decap_maps, decode_map, cap_objs_z, opt):
    decap_maps_z = [None] * len(decap_maps)
    decap_maps_holder = copy.deepcopy(decap_maps)

    for map_index,map in enumerate(decap_maps):   # for each list of decaps (chromosomes) in decap_maps
        decap_maps_z = []




def calc_impedance(decap_map, decode_map, opt):
    # decap_map = the list of encoded solutions
    # takes like a ~ 2.5 minutes to generate impedances and connected networks. For just 100 member population on lptp
    # takes like ~40 secs on my lab desktop
    solu_set = []
    zpdn = []
    for solu in decap_map: # each solution set in a population
        output_net = connect_n_decap(solu, decode_map, opt)
        solu_set.append(output_net)
        # Appends fully connected networks as they are created

        highest_z = np.amax(np.absolute(output_net.z[:,opt.ic_port-1,opt.ic_port-1]))
        zpdn.append(highest_z)



        #print('Max Impedance is:', highest_z, 'Ohms')
    return solu_set, zpdn





def testPlot(target_net,opt):
    # target_net is a network object (the board) connected to decaps
    for i in range(len(target_net)):
        print(i)
        plt.loglog(target_net[i].frequency.f,
               np.absolute(target_net[i].z[:, 0, 0]))

    plt.loglog(target_net[i].frequency.f, opt.ztarget)
    plt.grid(True,which = 'minor')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Impedance (Ohm)')
    plt.title('Impedance Curve vs Frequency')
    plt.legend(['Impedance Curve', 'Impedance Threshold, .015 Ohms'])
    plt.show()

def testPlot2(target_net_z,opt):
    # target_net is a network object (the board) connected to decaps
        #print(np.absolute(target_net_z[i,0,0]))
       # plt.loglog(input_net[i].frequency.f,
        #       np.absolute(target_net_z[i,0,0]))

    plt.loglog(opt.freq, np.absolute(target_net_z[:,0,0]))
    plt.loglog(opt.freq, opt.ztarget)
    plt.grid(True,which = 'minor')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Impedance (Ohm)')
    plt.title('Impedance Curve vs Frequency')
    plt.legend(['Impedance Curve', 'Impedance Threshold, .015 Ohms'])
    plt.show()





'''
def connect_n_decap(decap_map, decode_map, opt):
    target_net = opt.input_net  # asus board network object to connect to
    solu_net = []               
    for a in range(len(decap_map)):    # number of ports
        decaps_slln = decap_map[a]     # gives encoded number, 0 - 10, of cap to be placed
        for i in range(len(decaps_slln)):  # i is to give port location
            target_net = connect_1decap(target_net, decode_map[decaps_slln],i)
            # continually update the target_net as more cnncts
            # then need to calculate impedance before moving to next set
        solu_net.append(target_net)
    return solu_net   # this will return every resulting network for a set of solutions in decap_list
'''