# coding: utf-8
# Author: Ling Zhang
# Email : lzd76@mst.edu

import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


def find_minimum_freq_pt(opt, cut_off_f):
    # Find frequency point where you set minimum impedance
    min_pt = math.sqrt(opt.freq[0] * cut_off_f)
    return min_pt


def find_optimum_min(opt, cap_objs_z):
    cap_nums = range(1,opt.num_decaps+1)
    impedance_points = opt.freq
    optimum_min_pt = 0
    print('Maximum z target point is', np.amax(impedance_points))
    print('Minimum z target point is', np.amin(impedance_points))
    if np.amax(opt.ztarget) == np.amin(opt.ztarget):  # constant curve
        optimum_min_pt = np.ceil(opt.nf / 2).astype(np.int64)
        print('Minimum z should be around', opt.freq[optimum_min_pt], 'Hz')
    else:  # works for impedance curve changing once ( increasing, 27 port) for now only
        for index in range(opt.nf - 1):
            if opt.ztarget[index + 1] - opt.ztarget[index] != 0:
                optimum_min_pt = index
                print('Minimum z should be around', opt.freq[optimum_min_pt], 'Hz')
                break
    print('The optimum min index, based only on curve, is', optimum_min_pt)
    resonant_pts = [np.argmin((np.absolute(cap_objs_z[i][:, 0, 0]))) for i in range(opt.num_decaps)]
    shifted_resonant_pts = [abs(i - optimum_min_pt) for i in resonant_pts]
    print('Index of capacitor resonant frequencies:', resonant_pts)
    optimum_cap_point = resonant_pts[np.argmin(shifted_resonant_pts)]# can end up 0 - 1 = -1
    next_cap_point = resonant_pts[np.argmin(shifted_resonant_pts) - 1]
    print('The decap whose resonant index is immediately larger is:', np.argmin(resonant_pts) - 1)
    print('With its resonant frequency at index', next_cap_point, 'at', opt.freq[next_cap_point], 'Hz')
    print('Min z of optimal solution should be around', opt.freq[next_cap_point], 'Hz')
    return optimum_min_pt


def target_shift(opt):
    # works for impedance curve changing once only
    # Could pull this off the text file directly but it works

    shift_pt = []
    print('Maximum z target point is', np.amax(opt.freq))
    print('Minimum z target point is', np.amin(opt.freq))


    if np.amax(opt.ztarget) == np.amin(opt.ztarget):
        # constant curve, zero slope
        print('Target Curve is Constant, target shift point set as last point')
        shift_pt = opt.freq[-1]

    else:
        for index in range(opt.nf - 1):
            if opt.ztarget[index + 1] - opt.ztarget[index] != 0:
                shift_pt = opt.freq[index]
                print('Shift Point Occurs at index:', index, 'with z =', shift_pt)
    return shift_pt



def find_maxs(decap_map_z, local_only = False):
    map_z = np.absolute(decap_map_z)
    holder = copy.deepcopy(map_z)
    maxs = []

    for index, z in enumerate(holder):
        if index == (len(holder) - 1) or index == 0:  # disregarding endpoints
            pass
        elif holder[index - 1] < z and z > holder[index + 1]:
            maxs.append(index)
        else:
            pass
    if local_only:
        for i in maxs:
            if i == np.argmax(map_z):
                maxs.remove(i)  # remove the global min value
    return maxs




def find_mins(decap_map_z, local_only = False):
    map_z = np.absolute(decap_map_z)
    holder = copy.deepcopy(map_z)
    mins = []
    for index, z in enumerate(holder):
        if index == (len(holder) - 1) or index == 0: # disregarding endpoints
            pass
        elif holder[index - 1] > z and z < holder[index + 1]:
            mins.append(index)
        else:
            pass
    if local_only:
        for i in mins:
            if i == np.argmin(map_z):
                mins.remove(i) # remove the global min value
    return mins

def short_1port(input_net, shorted_port=2):
    # default shorted port for decap is port 2. if input_net is a network, need to point out shorted port #
    short_net = copy.deepcopy(input_net.s11)
    short_net.s = -1*np.ones(short_net.f.shape[0])
    output_net = rf.network.connect(input_net, shorted_port-1, short_net, 0)
    return output_net


def select_decap(decap_num, opt):
    decap = []
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

def calc_z(decap_maps, cap_objs_z, board_z11):
    decap_maps_z = [None] * len(decap_maps)
    z_inv = 0
    for map_num, decaps in enumerate(decap_maps):
        for decap in decaps:
            z_inv = z_inv + 1/(np.absolute(cap_objs_z[decap - 1][:,0,0])) if decap != 0 else z_inv + 0
        z_inv = z_inv + 1/(np.absolute(board_z11))   # add impedance seen by port 1
        decap_maps_z[map_num] = np.reciprocal(copy.deepcopy(z_inv))
        z_inv = 0
    return decap_maps_z

def new_connect_1decap(input_net_z, connect_port, decap_z11):
    output_net_z = new_connect(input_net_z, connect_port, decap_z11)
    return output_net_z

def new_connect_n_decap(input_net_z, decap_map, cap_objs_z, opt):
    # formally used to connect decaps
    decap_map_z = copy.deepcopy(input_net_z)
    port_num = opt.ic_port   # Has the problem where this only works if IC port is the very first port (index 0).
                             # Okay for now but need generalize

    for decap in decap_map:
        if decap != 0:
            decap_map_z = new_connect_1decap(decap_map_z, port_num, cap_objs_z[decap-1])
        else:
            port_num = port_num + 1
    decap_map_z = copy.deepcopy(decap_map_z[:,0,0])
    return decap_map_z

def connect_networks(decap_maps, cap_objs_z, opt):
    decap_maps_z = []
    for decap_map in decap_maps:
        decap_maps_z = [new_connect_n_decap(opt.input_net.z,decap_map, cap_objs_z, opt) for i in range(len(decap_maps))]
    return decap_maps_z

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


def final_check(min_zero_map,opt,cap_objs_z):

    improve_bool = True
    while improve_bool is True:
        current_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # number of caps in the initial min_zero_map
        print('Before check, decap map is:', min_zero_map)
        for ind, _ in enumerate(min_zero_map): # iterating through each decap in min map
            holder = copy.deepcopy(min_zero_map) # make copy

            if min_zero_map[ind] != 0: # if port is not empty
                holder[ind] = 0  # make port empty
                holder_z = [new_connect_n_decap(opt.input_net.z, holder, cap_objs_z, opt)]
                if np.count_nonzero(np.greater(np.absolute(holder_z), opt.ztarget)) == 0:
                    # if # of capacitors decrease and target met, overwrite min zero map
                    min_zero_map = copy.deepcopy(holder) # update to better map
                    # improve_bool still true
                    break
                else:
                    holder = copy.deepcopy(min_zero_map)
                    # if target impedance not met, recapture min_zero_map
                    # and set the next non-empty port 0
        new_min = len(min_zero_map) - np.count_nonzero(min_zero_map) # used to set improve bool
        if new_min > current_min:
            print('After check, number of capacitors decreased. Min decap layout is', min_zero_map)
            print('Checking Again....')
            improve_bool = True # not needed but helps me with clarity
        else:
            print('After check, number of capacitors did not decrease.')
            improve_bool = False # score did not improve, set improve_bool to false. break out of loop

    return min_zero_map

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

def new_test_plot(z_to_plot,opt):
    # z_to_plot is a list of lists of the z to be plotted
    # for now written to plot min decap and best scoring
    for i in z_to_plot:
        plt.loglog(opt.freq, i)
    plt.loglog(opt.freq, opt.ztarget)

    # I was trying to check something
    #plt.plot((opt.freq[66],opt.freq[66]), (0,opt.ztarget[66]),'k-')
    #plt.plot((opt.freq[133], opt.freq[133]), (0, opt.ztarget[133]), 'k-')
    #plt.plot((opt.freq[100], opt.freq[100]), (0, opt.ztarget[100]), 'k-')


    plt.grid(True,which = 'minor')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Impedance (Ohm)')
    plt.title('Impedance Curve vs Frequency')
    plt.legend(['Min Decap Curve', 'Best Scoring Curve', 'Impedance Target'])
    plt.show()


def new_test_plot2(z_to_plot,opt):
    # z_to_plot is a list of lists of the z to be plotted
    for i in z_to_plot:
        plt.loglog(opt.freq, i)
    plt.loglog(opt.freq, opt.ztarget)

    plt.grid(True, which='minor')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Impedance (Ohm)')
    plt.title('Impedance Curve vs Frequency')
    plt.legend([i+1 for i in range(len(z_to_plot))] + ['Impedance Target'])
    #plt.legend(['Min Decap Curve', 'Best Scoring Curve', 'Impedance Target'])
    plt.show()



