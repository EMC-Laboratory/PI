# coding: utf-8
# Author: Ling Zhang
# Email : lzd76@mst.edu

import skrf as rf
import copy
import numpy as np
from math import pi


def merge_ports(orig_net, map2orig_input, ic_port_list, used_ic_port):
    # ic_port_list, used_ic_port index begins with 1
    orig_port_list = list(range(1, orig_net.s.shape[1]+1))
    del_ic_port = copy.deepcopy(ic_port_list)
    del_ic_port.remove(used_ic_port)
    left_port_list = copy.deepcopy(orig_port_list)

    for a in del_ic_port:
        left_port_list.remove(a)

    map2orig_output = [map2orig_input[i-1] for i in left_port_list]

    for a in range(len(left_port_list)):
        left_port_list[a] = left_port_list[a] - 1

    # calculate inverse z matrix
    z_inv = np.linalg.inv(orig_net.z)
    z_inv_merge = copy.deepcopy(z_inv)

    # add up inverse z matrix rows and columns
    for i in range(0, orig_net.z.shape[0]):  # sweep frequency
        for j in del_ic_port:
            z_inv_merge[i, used_ic_port - 1, :] = z_inv_merge[i, used_ic_port - 1, :] + z_inv_merge[i, j - 1, :]
        for k in del_ic_port:
            z_inv_merge[i, :, used_ic_port - 1] = z_inv_merge[i, :, used_ic_port - 1] + z_inv_merge[i, :, k - 1]

    z_inv_merge = z_inv_merge[0:orig_net.s.shape[0], left_port_list, :][0:orig_net.s.shape[0], :, left_port_list]

    z_merge = np.linalg.inv(z_inv_merge)

    # convert back to s-parameters, type numpy array
    s_merge = rf.network.z2s(z_merge)

    merged_net = copy.deepcopy(orig_net)
    merged_net.s = s_merge
    merged_net.frequency = orig_net.frequency
    merged_net.z0 = 50

    # port_map_orig is the port map to the original port number
    return merged_net, map2orig_output


def short_1port(input_net, map2orig_input=[1, 2], shorted_port=2):
    # default shorted port for decap is port 2. if input_net is a network, need to point out shorted port #
    short_net = copy.deepcopy(input_net.s11)
    short_net.s = -1*np.ones(short_net.f.shape[0])
    output_net = rf.network.connect(input_net, shorted_port-1, short_net, 0)

    map2orig_output = copy.deepcopy(map2orig_input)
    del map2orig_output[shorted_port-1]

    return output_net, map2orig_output


def short_nport(input_net, map2orig_input, shorted_ports):
    output_net = copy.deepcopy(input_net)
    map2orig_output = list(range(1, output_net.s.shape[1]+1))

    for a in shorted_ports:
        output_net, map2orig_output = short_1port(output_net, map2orig_output,
                                                  shorted_port=map2orig_output.index(a)+1)

    map2orig_output = [map2orig_input[i-1] for i in map2orig_output]
    return output_net, map2orig_output


def connect_1decap(input_net, map2orig_input, connect_port, decap_s11):
    output_net = rf.network.connect(input_net, connect_port-1, decap_s11, 0)
    map2orig_output = copy.deepcopy(map2orig_input)
    del map2orig_output[connect_port-1]
    return output_net, map2orig_output


def connect_n_decap(input_net, map2orig_input, connect_ports, decap_list, opt):
    output_net = copy.deepcopy(input_net)
    map2orig_output = list(range(1, output_net.s.shape[1] + 1))

    for a in range(len(connect_ports)):
        output_net, map2orig_output = connect_1decap(output_net, map2orig_output,
                                                     connect_port=map2orig_output.index(connect_ports[a])+1,
                                                     decap_s11=select_decap(decap_list[a], opt))

    map2orig_output = [map2orig_input[i - 1] for i in map2orig_output]

    return output_net, map2orig_output


# prioritize ports by shorting ports respectively and looking at inductance of Z11. Shorting multiple times
def prioritize_ports_2(input_net, map2orig_input, ic_port=1, start_freq=10, stop_freq=20):
    # ic_port only has one port
    # start_freq, stop_freq: start and stop frequency of linear inductance region. Unit MHz
    freq_l = rf.frequency.Frequency(start=start_freq, stop=stop_freq, npoints=201, unit='mhz', sweep_type='log')
    orig_port = list(range(1, input_net.s.shape[1] + 1))
    compare_ports = list(range(1, input_net.s.shape[1] + 1))  # ports that need to be prioritized
    compare_ports.remove(ic_port)
    compare_ports_orig = [map2orig_input[i - 1] for i in compare_ports]  # corresponding to original port number

    port_sort = []
    port_map = copy.deepcopy(orig_port)

    for a in range(1, len(compare_ports)+1):
        l11s = []
        for b in compare_ports:
            net_tmp, port_map_tmp = short_1port(input_net, map2orig_input=port_map, shorted_port=port_map.index(b)+1)
            l11s.append(calcu_l11_thru_phase(net_tmp, port1=port_map_tmp.index(ic_port)+1, freq_l=freq_l))

        port_sort.append(compare_ports[l11s.index(min(l11s))])
        input_net, port_map = short_1port(input_net, map2orig_input=port_map,
                                          shorted_port=port_map.index(compare_ports[l11s.index(min(l11s))])+1)
        del compare_ports[l11s.index(min(l11s))]
        del compare_ports_orig[l11s.index(min(l11s))]
    map2orig_output = [map2orig_input[i - 1] for i in port_sort]

    return port_sort, map2orig_output


# prioritize ports by shorting ports respectively and looking at inductance of Z11. Shorting only one time
def prioritize_ports_3(input_net, map2orig_input, ic_port=1, start_freq=10, stop_freq=20):
    # ic_port only has one port
    # start_freq, stop_freq: start and stop frequency of linear inductance region. Unit MHz
    freq_l = rf.frequency.Frequency(start=start_freq, stop=stop_freq, npoints=201, unit='mhz', sweep_type='log')
    orig_port = list(range(1, input_net.s.shape[1] + 1))
    compare_ports = list(range(1, input_net.s.shape[1] + 1))  # ports that need to be prioritized
    compare_ports.remove(ic_port)
    compare_ports_orig = [map2orig_input[i - 1] for i in compare_ports]  # corresponding to original port number

    port_sort = []
    port_map = copy.deepcopy(orig_port)

    l11s = []
    for b in compare_ports:
        net_tmp, port_map_tmp = short_1port(input_net, map2orig_input=port_map, shorted_port=port_map.index(b)+1)
        l11s.append(calcu_l11_thru_phase(net_tmp, port1=port_map_tmp.index(ic_port)+1, freq_l=freq_l))

    port_sort = [compare_ports[i] for i in sorted(range(len(l11s)), key=lambda k: l11s[k])]

    map2orig_output = [map2orig_input[i - 1] for i in port_sort]

    return port_sort, map2orig_output


def prioritize_ports(input_net, map2orig_input, ic_port=1, start_freq=10, stop_freq=20, option='phase'):
    # ic_port only has one port
    # start_freq, stop_freq: start and stop frequency of linear inductance region. Unit MHz
    freq_l = rf.frequency.Frequency(start=start_freq, stop=stop_freq, npoints=201, unit='mhz', sweep_type='log')
    orig_port = list(range(1, input_net.s.shape[1] + 1))
    compare_ports = list(range(1, input_net.s.shape[1] + 1))            # ports that need to be prioritized
    compare_ports.remove(ic_port)
    compare_ports_orig = [map2orig_input[i-1] for i in compare_ports]   # corresponding to original port number

    port_sort = []
    port_map = copy.deepcopy(orig_port)

    for a in range(1, len(compare_ports)+1):
        l1n_l1n_lnn = []
        for b in compare_ports:
            if option == 'phase':
                l1n_l1n_lnn.append(calcu_l1n_l1n_lnn_thru_phase(input_net, port1=ic_port, portn=port_map.index(b)+1,
                                                           freq_l=freq_l))
            elif option == 'select':
                l1n_l1n_lnn.append(calcu_l1n_l1n_lnn_select(input_net, port1=ic_port, portn=port_map.index(b)+1,
                                                            freq_l=freq_l))
        port_sort.append(compare_ports[l1n_l1n_lnn.index(max(l1n_l1n_lnn))])
        input_net, port_map = short_1port(input_net, map2orig_input=port_map,
                                          shorted_port=port_map.index(compare_ports[l1n_l1n_lnn.index(max(l1n_l1n_lnn))])+1)
        del compare_ports[l1n_l1n_lnn.index(max(l1n_l1n_lnn))]
        del compare_ports_orig[l1n_l1n_lnn.index(max(l1n_l1n_lnn))]
    map2orig_output = [map2orig_input[i - 1] for i in port_sort]

    return port_sort, map2orig_output


# calculate inductance by defining a frequency region and use least square method
def calcu_l1n_l1n_lnn_select(input_net, port1, portn, freq_l):
    lnn = np.linalg.lstsq(np.reshape(2 * pi * freq_l.f, (freq_l.f.shape[0], 1)),
                          np.reshape(np.absolute(input_net.interpolate(freq_l).z[:, portn - 1, portn - 1]),
                                     (freq_l.f.shape[0], 1)), rcond=None)[0][0][0]
    l1n = np.linalg.lstsq(np.reshape(2 * pi * freq_l.f, (freq_l.f.shape[0], 1)),
                          np.reshape(np.absolute(input_net.interpolate(freq_l).z[:, port1 - 1, portn - 1]),
                                     (freq_l.f.shape[0], 1)), rcond=None)[0][0][0]
    return l1n*l1n/lnn


# calculate inductance by looking for a 90 degree phase point. This method is not reliable for now
def calcu_l1n_l1n_lnn_thru_phase(input_net, port1, portn, freq_l):
    lnn_induct_index = np.argmin(np.absolute(np.angle(input_net.interpolate(freq_l).z[:, portn-1, portn-1])-pi/2))
    lnn = np.imag(input_net.interpolate(freq_l).z[lnn_induct_index, portn-1, portn-1])/(input_net.interpolate(freq_l).frequency.f[lnn_induct_index]*2*pi)

    l1n_induct_index = np.argmin(np.absolute(np.angle(input_net.interpolate(freq_l).z[:, port1 - 1, portn - 1]) - pi / 2))
    l1n = np.imag(input_net.interpolate(freq_l).z[l1n_induct_index, port1 - 1, portn - 1]) / (input_net.interpolate(freq_l).frequency.f[l1n_induct_index]*2*pi)

    return l1n*l1n/lnn


def calcu_l11_thru_phase(input_net, port1, freq_l):
    l11_induct_index = np.argmin(
        np.absolute(np.angle(input_net.interpolate(freq_l).z[:, port1 - 1, port1 - 1]) - pi / 2))
    l11 = np.imag(input_net.interpolate(freq_l).z[l11_induct_index, port1 - 1, port1 - 1]) / (
                input_net.interpolate(freq_l).frequency.f[l11_induct_index] * 2 * pi)
    return l11


'''def select_decap(decap_num, opt):
    if decap_num == 1:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM1553U1A272JA01.s2p').interpolate(opt.Freq))
    elif decap_num == 2:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM1557U1A332JA01.s2p').interpolate(opt.Freq))
    elif decap_num == 3:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM1557U1A392JA01.s2p').interpolate(opt.Freq))
    elif decap_num == 4:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM1553U1A472JA01.s2p').interpolate(opt.Freq))
    elif decap_num == 5:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM15XB11C103KA86.s2p').interpolate(opt.Freq))
    elif decap_num == 6:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM155R61H473ME19.s2p').interpolate(opt.Freq))
    elif decap_num == 7:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM152B31A104ME19.s2p').interpolate(opt.Freq))
    elif decap_num == 8:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM152B30J474ME15.s2p').interpolate(opt.Freq))
    elif decap_num == 9:
        decap, _ = short_1port(rf.Network('decap/decap_s2p/GRM153R60G105ME95.s2p').interpolate(opt.Freq))
    return decap'''


def select_decap(decap_num, opt):
    if decap_num == 1:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM033C80J104KE84.s2p').interpolate(opt.Freq))
    elif decap_num == 2:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM033R60J474KE90.s2p').interpolate(opt.Freq))
    elif decap_num == 3:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM155B31C105KA12.s2p').interpolate(opt.Freq))
    elif decap_num == 4:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM155C70J225KE11.s2p').interpolate(opt.Freq))
    elif decap_num == 5:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM185C81A475KE11.s2p').interpolate(opt.Freq))
    elif decap_num == 6:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM188R61A106KAAL.s2p').interpolate(opt.Freq))
    elif decap_num == 7:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM188B30J226MEA0.s2p').interpolate(opt.Freq))
    elif decap_num == 8:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM219D80E476ME44.s2p').interpolate(opt.Freq))
    elif decap_num == 9:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM31CR60J227ME11.s2p').interpolate(opt.Freq))
    elif decap_num == 10:
        decap, _ = short_1port(rf.Network('decap/decap_s2p_v2/GRM32EC80E337ME05.s2p').interpolate(opt.Freq))
    return decap
