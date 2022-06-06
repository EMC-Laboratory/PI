import PopInit
import pdn
from config2 import Config
import GenFlow
import numpy as np
import datetime
import random
import copy
import pdn

def OptionsInit():
    # Get settings
    opt = Config()
    return opt


def get_initial_pop(Pop, opt):
    decap_map = copy.deepcopy(Pop.chromosomes)   # Gets a copy of the population
    cap_objs = [pdn.select_decap(i, opt) for i in range(Pop.numDecaps)] # list of shorted capacitors
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(1,opt.num_decaps)]
    return decap_map, cap_objs, cap_objs_z

#
# def get_low_freq_cutoff(opt):
#     break_freq = break_freq_index = 0
#     for i in range(opt.nf):
#         if abs(opt.input_net.z[i,opt.ic_port-1,opt.ic_port-1]) > opt.ztarget[i]:
#             break_freq = opt.freq[i]
#             break_freq_index = i
#             print('Break Frequency set at', break_freq, 'Hz')
#             break
#     if break_freq == 0:
#         raise ValueError('Without any decaps, the impedance profile is already met!')
#     return break_freq, break_freq_index



if __name__ == '__main__':

    # Intialization steps
    opt = OptionsInit()  # Set some settings
    input_net = copy.deepcopy(opt.input_net)
    input_net_z = copy.deepcopy(opt.input_net.z)
    Population = PopInit.Population(opt)  # Create a Population object
    Population.create_population(opt)     # Create initial population
    decap_maps, cap_objs, cap_objs_z = get_initial_pop(Population, opt) # Gets population layout and the 'decoder' that maps # to a decap
    #break_freq, break_freq_index = get_low_freq_cutoff(opt)  # get break frequency index
    decap_map_z_list = [] # holder for z parameters of all layouts in population

    for decap_map in decap_maps:
        decap_map_z = pdn.new_connect_n_decap(input_net_z, decap_map, cap_objs_z, opt)
        decap_map_z_list.append(decap_map_z)
    decap_maps_impedance = [np.absolute(decap_map_z_list[i][:,0,0]) for i in range(len(decap_map_z_list))]
    #Get impedance matrix for all frequencies for all layouts in population

