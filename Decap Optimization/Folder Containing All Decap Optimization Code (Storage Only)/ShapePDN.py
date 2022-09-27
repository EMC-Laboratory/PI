# coding: utf-8
# Author: Ling Zhang
# Email : lzd76@mst.edu

import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from numba import jit
import time
def calc_first_antires(decap_map, cap_objs_z, opt):
    # his function will take a map/solution with ONLY 1 capacitor in it
    # it will calculate the theoretical Ceq associated with that anti res peak
    # not to useful if peak is below target though
    lvrm = 2 * pow(10,-9)
    rvrm = 2 * pow(10,-3)

    decap_map_z = new_connect_n_decap(opt.input_net.z, decap_map,cap_objs_z, opt)
    print(decap_map_z)
    local_maxs = find_maxs(decap_map_z)
    print(local_maxs[0])
    print(decap_map_z[local_maxs[0]])
    plt.loglog(opt.freq, np.abs(decap_map_z))
    plt.show()
    ref_freq_ind = local_maxs[0]


#(0.08471306592088262+0.017615300699731357j)



def rl_target_change(shift_f, opt):
    # this function will create a new z target for RL shape targets if the resulting slope is not 20 db per decade

    # I THINK THIS IS WORKING

    # Stuff to return (uneeded but eh)
    old_ztarget = copy.deepcopy(opt.ztarget)
    old_freq = copy.deepcopy(opt.freq)
    new_shift_f = shift_f

    decades = math.log10(opt.freq[-1]/shift_f)
    shift_index = np.nonzero(opt.freq == shift_f)[0]
    gain = 20 * decades  # 20 dB per decade, expected db gain

    exp_z = math.pow(10, gain / 20) * opt.ztarget[shift_index]
    # expected z of last point if slope changes at 20 dB per decade
    # if actual last z is lower, slope is less than 20 dB per decade,
    # if actual last z is higher, slope is greater than 20 dB per decade
    print('Old frequency point where slope changes is:', shift_f)
    # If target z does not change (constant R):
    if shift_f == opt.freq[-1]:
        print('Target is of constant R type (constant slope). Target impedance will not need to be changed')

    elif opt.ztarget[-1] >=  exp_z:
        print('Target is of RL type, but the slope is >= than 20 dB per decade. It does not need to be changed')

    elif opt.ztarget[-1] < exp_z:
        # This function will make the slope change happen later (target more stringent) in order to get
        # a 20 dB curve. It SHOULD work. In theory. Might not work with the current min cap selection but yolo
        # final z point is the same.

        print('Target is of RL type, but the slope is < than 20 dB per decade, adjusting target so slope is 20dB/dec')
        gain = 20*math.log10(opt.ztarget[-1]/opt.ztarget[shift_index])
        # this is the gain you should have if it was 20 db per decade slope
        # and that gain = 20 * log10( final f/ new shift f) new shift is what we want
        new_shift_f = opt.freq[-1] / math.pow(10, gain/20)

        old_ztarget = copy.deepcopy(opt.ztarget)
        old_freq = copy.deepcopy(opt.freq)
        start_z = old_ztarget[0]
        last_z = old_ztarget[-1]

        new_ztarget_file = np.asarray([[opt.freq[0]/1e6, start_z], [new_shift_f/1e6, start_z], [opt.freq[-1]/1e6, last_z]])

        opt.ztarget_file = copy.deepcopy(new_ztarget_file)

        opt.freq = np.logspace(np.log10(opt.fstart), np.log10(opt.fstop), opt.nf)
        opt.ztarget = np.interp(opt.freq, opt.ztarget_file[:, 0] * 1e6, opt.ztarget_file[:, 1])

    print('New frequency point where slope changes is (Maybe unchanged):', new_shift_f)
    return old_freq, old_ztarget
def find_minimum_freq_pt(opt, cut_off_f):
    # Find frequency point where you set minimum impedance
    min_pt = math.sqrt(opt.freq[0] * cut_off_f)
    return min_pt


def calculate_rough_distance(opt,cap_objs_z):
    holder = [0] * opt.decap_ports
    decap_maps = [copy.deepcopy(holder) for i in range(opt.decap_ports)]

    j = 0
    for map in decap_maps:
        map[j] = 10
        j = j + 1
    holderz = [new_connect_n_decap(opt.input_net.z, decap_maps[i], cap_objs_z, opt) for i in
               range(opt.decap_ports)]
    max_z_pt = [[i + 1, (np.abs(holderz[i][-1]))] for i in range(opt.decap_ports)]
    max_z_pt = sorted(dict(max_z_pt).items(), key=lambda x: x[1])
    distances = [i[0] for i in max_z_pt]
    return distances

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

    # Could pull this off the text file directly but it works

    shift_pt = []


    if np.amax(opt.ztarget) == np.amin(opt.ztarget):
        # constant curve, zero slope
        print('Target Curve is Constant, target shift point set as last point')
        shift_pt = opt.freq[-1]

    else:
        for index in range(opt.nf - 1):
            if opt.ztarget[index + 1] - opt.ztarget[index] != 0:
                shift_pt = opt.freq[index]
                print('Point where slope changes occurs at', index, 'with f =', shift_pt)
                break
    return shift_pt



def find_maxs(decap_map_z, local_only = False, last_z = True):
    # returns the indices of local maximums
    map_z = np.absolute(decap_map_z)
    holder = copy.deepcopy(map_z)
    maxs = []

    for index, z in enumerate(holder):
        if index == 0:  # don't include first point as a possible local max
            pass
        elif index == (len(holder) - 1) and holder[index] > holder[index-1] and last_z:
            # Include last point as a maximum
            maxs.append(index)
        elif index == (len(holder)-1) and not last_z:
            # If last point is not to be considered a local max
            pass
        elif holder[index - 1] < z and z > holder[index + 1]:
            maxs.append(index)

    if local_only:
        max_z_arg = np.argmax(map_z) # fix to one value, so it doesn't vary around in loop. Or just use break
        for i in maxs:
            if i == max_z_arg:
                maxs.remove(i)  # remove the global max value
    return maxs




def find_mins(decap_map_z, local_only = False):

    # returns the indices containing the local minimums
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
        min_z_arg = np.argmin(map_z)  # fix to one value, so it doesn't vary around in loop. Or just use break
        for i in mins:
            if i == min_z_arg:
                mins.remove(i) # remove the global min value
    return mins

def find_min_cap(shift_f,cap_objs_z,initial_z,opt):
    found = False

    decap_maps = [[0] * opt.decap_ports for i in range(1, opt.num_decaps + 1)]
    iter_index = 0

    j = 10  # starting with 10 because this should be easier/faster than reversing the list multiple times
            # Less clear though
            # Reason for flipping is so you guarentee the largest small capacitor
    min_cap = []
    ref = 0
    holder = []

    while not found:
        # Create maps

        for x, y in enumerate(decap_maps):
            decap_maps[x][iter_index] = j
            j = j - 1
        iter_index = iter_index + 1
        j = 10

        # Calculate impedance for maps
        decap_maps_z = [new_connect_n_decap(initial_z, decap_maps[i], cap_objs_z, opt) for i in
                            range(opt.num_decaps)]

        for cap, z in enumerate(decap_maps_z):
            # get local minimum (or set as last point
            local_mins = find_mins(z,local_only= True)

            if len(local_mins) == 0:
                local_mins = [opt.nf - 1]

            ref = local_mins[-1]

            if opt.freq[ref] <= shift_f and np.abs(z)[ref] <= opt.ztarget[ref]:
                # if last local min is behind the frequency where the slope changes, and meets the impedance at that
                # pt

                min_cap = decap_maps[cap][0]
                holder.append([min_cap, ref])

        for i in holder:

            if i[1] == opt.nf-1:

                holder.remove(i)
        print(decap_maps)
        print(holder)
        if len(holder) != 0:
            min_cap = holder[-1][0]
            found = True
            break


        if iter_index == len(decap_maps[0]) and not found:
            raise ValueError('No solutions exist that can satisfy the target impedance')
    return ref, min_cap

def find_min_cap2(bulk_cap, shift_f,cap_objs_z,initial_z,opt):
    found = False

    decap_maps = [[0] * opt.decap_ports for i in range(1, opt.num_decaps + 1)]
    iter_index = 0

    j = 10  # starting with 10 because this should be easier/faster than reversing the list multiple times
    # Less clear though
    # Reason for flipping is so you guarentee the largest small capacitor
    min_cap = []
    ref = 0

    while not found:
        # Create maps

        for x, y in enumerate(decap_maps):
            decap_maps[x][iter_index] = j
            j = j - 1
        iter_index = iter_index + 1
        j = 10

        # Calculate impedance for maps
        decap_maps_z = [new_connect_n_decap(initial_z, decap_maps[i], cap_objs_z, opt) for i in
                        range(opt.num_decaps)]

        for cap, z in enumerate(decap_maps_z):
            # get local minimum (or set as last point
            local_mins = find_mins(z, local_only=True)

            if len(local_mins) == 0:
                local_mins = [opt.nf - 1]

            ref = local_mins[-1]

            if np.abs(z)[-1] <= opt.ztarget[-1] and opt.freq[ref] <= shift_f and np.abs(z)[ref] <= opt.ztarget[ref]:
                # if last target point is satisfied and last local min (or last z point if local SRF outside),
                # is beneath and behhind the shift pt, then pick that capacitor to be the min capacitor
                min_cap = decap_maps[cap][0]
                found = True
                break



            # if opt.freq[ref] <= shift_f and np.abs(z)[ref] <= opt.ztarget[ref]:
            #     # if last SRF is before where slope changes, and SRF meets target
            #     min_cap = decap_maps[cap][0]
            #     found = True
            #     break

        if iter_index == len(decap_maps[0]) and not found:
            min_cap = 1
            raise ValueError('No solutions exist that can satisfy the target impedance')
    return ref, min_cap



def find_bulk_cap(shift_f,cap_objs_z,initial_z,opt):


    found = False
    decap_maps = [[0] * opt.decap_ports for i in range(1, opt.num_decaps + 1)]
    iter_index = 0

    j = 1
    bulk_cap = []
    ref = 0
    ref2 = 0
    while not found:
        # Create maps

        for x, y in enumerate(decap_maps):
            decap_maps[x][iter_index] = j
            j = j + 1
        iter_index = iter_index + 1
        j = 1

        decap_maps_z = [new_connect_n_decap(initial_z, decap_maps[i], cap_objs_z, opt) for i in
                            range(opt.num_decaps)]

        for cap, z in enumerate(decap_maps_z):
            # get local minimum (or set as last point
            local_maxs = find_maxs(z,local_only= False, last_z= False)
            local_mins = find_mins(z,local_only= False)

            if len(local_maxs) == 0:
                local_maxs = [opt.nf - 1]

            if len(local_mins) == 0:
                local_mins = [opt.nf - 1]

            ref = local_maxs[0]
            ref2 = local_mins[0]
            if np.abs(z)[ref] <= opt.ztarget[ref] and opt.freq[ref] <= shift_f:
                # If the first anti resonant peak is less than or equal to the target
                # and before where slope changes, set as bulk cap
                bulk_cap = decap_maps[cap][0]
                found = True
                break

        if iter_index == len(decap_maps[0]) and not found:
            raise ValueError('No solutions exist that can satisfy the target impedance')
    return ref2, bulk_cap




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

def ideal_calc_z(decap_maps, cap_objs_z, board_z11):
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

    #decap_map_z = copy.deepcopy(input_net_z)
    decap_map_z = input_net_z.copy()

    port_num = opt.ic_port   # Has the problem where this only works if IC port is the very first port (index 0).
                             # Okay for now, can just re-number ports with some other written function

                             #Port num is assumed port 1 in the s parameters,the first port to fill is therefore port 2.
                             # However decap_map is written to include only ports where you put decaps. So index 0 of the
                             # decap_map, corresponds to port 2 of the sparameters. Ie if opt.ic_port = 1, this function
                             # will work because we start filling index 1 of the input z array (port 2 of s parameters)

    for decap in decap_map:
        if decap != 0:
           # decap_map_z = new_connect_1decap(decap_map_z, port_num, cap_objs_z[decap-1])
            decap_map_z = new_connect(decap_map_z, port_num, cap_objs_z[decap-1])
        else:
            port_num = port_num + 1
    decap_map_z = decap_map_z[:,0,0].copy()

    return decap_map_z

def connect_networks(decap_maps, cap_objs_z, opt):
    decap_maps_z = []
    for decap_map in decap_maps:
        decap_maps_z = [new_connect_n_decap(opt.input_net.z,decap_map, cap_objs_z, opt) for i in range(len(decap_maps))]
    return decap_maps_z


def new_connect(input_net_z, connect_port, decap_z11):  # or arbitrary z11

    Zaa = np.copy(input_net_z)
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
    #inv = np.linalg.solve(Zpp+Zqq, np.ones(np.shape(Zpp)))
    #inv = 1/(Zpp + Zqq) # inv was of dimension [201,1,1] so it would work to just take 1/x


    second = np.einsum('rmn,rkk->rmn', Zap, inv, optimize= 'True')
    second = np.einsum('rmn,rnd->rmd', second, Zpa, optimize = 'True')

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



