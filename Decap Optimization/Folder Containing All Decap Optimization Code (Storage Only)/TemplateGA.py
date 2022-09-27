from config2 import Config
import ShapePDN as pdn1
import ShapeGAOps as GA1
import copy
import numpy as np
import PopInit as Pop
import time
import math
import random
from matplotlib import pyplot as plt

def OptionsInit():
    # Get settings
    opt = Config()
    return opt


def decap_objects(opt):
    # Store capacitor library as their impedances
    cap_objs = [pdn1.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors, 1 to 10, high end is not inclusive [x,y) so need + 1
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z



def get_target_z_RL(R, Zmax, fstart=1e4, fstop=20e6, nf=201, interp='log'):
    freq  = 0
    f_transit = fstop * R / Zmax
    if interp =='log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return ztarget




if __name__ == '__main__':

    ########## Set Initial Settings ##############
    start_time = time.time()
    opt = OptionsInit()                             # Initialize Run Settings
    cap_objs, cap_objs_z = decap_objects(opt)       # Initialize Capacitor Impedances
    Population = Pop.Population(opt)                # Create Population object for GA
    Population.create_population(opt)               # Create Population for GA
    decap_maps = Population.return_chromosomes()    # Store Population for use by GA
    initial_z = np.absolute(opt.input_net.z[:,0,0]) # Get Impedance of Board
    initial_sol_found = False
    initial_sol_index = []
    best_map = []                                 # Variable to store best scoring sol
    best_map_z = []
    best_score = 0                                  # Variable to store best score
    best_z = []
    min_zero_map = []                               # Variable to store minimum decap solution
    min_zero_index = []
    post_decap_maps_scores = []



    ######
    R = .01
    Zmax = .02
    opt.ztarget = get_target_z_RL(R, Zmax, fstart=1e4, fstop=20e6, nf=201, interp='log')

    ####### Begin GA ######
    while Population.current_gen <= Population.generations:

        print('Generation', Population.current_gen, 'Start!')
        #print('Population:', decap_maps)
        # Calculate z
        decap_maps_z = [pdn1.new_connect_n_decap(opt.input_net.z,decap_maps[i],cap_objs_z,opt) for i in range(Population.population)]

        # Calculate scores and look for solutions in initial population
        decap_maps_scores = GA1.basic_score(decap_maps, decap_maps_z, opt)
        print('Current Scores:', decap_maps_scores)

        if max(decap_maps_scores) > best_score:
            best_score = max(decap_maps_scores)
            best_index = decap_maps_scores.index(best_score)
            best_map = copy.deepcopy(decap_maps[best_index])
            best_map_z = copy.deepcopy(decap_maps_z[best_index])
            print('Score Improved. Best Scores:', best_score, 'map', best_map)

        ########## Begin GA operations ########
        # Select parents for crossover based on score
        selected_parents = GA1.roulette(decap_maps_scores, decap_maps)

        cross_population = GA1.cross_over(selected_parents, Population, opt)

        mutated_population = GA1.mutation2(Population, cross_population, opt)

        decap_maps = copy.deepcopy(mutated_population)

        # Recalculate scores after GA operators of the final generation
        if Population.current_gen == Population.generations:

            #print('Final Population:', decap_maps)


            decap_maps_z = [pdn1.new_connect_n_decap(opt.input_net.z, decap_maps[i], cap_objs_z, opt) for i in
                            range(Population.population)]
            decap_maps_scores = GA1.basic_score(decap_maps, decap_maps_z, opt)

            print('Final Scores:', decap_maps_scores)

            if max(decap_maps_scores) > best_score:
                print('Score Improved. Current Scores:', decap_maps_scores)
                best_score = max(decap_maps_scores)
                best_index = decap_maps_scores.index(best_score)
                best_map = copy.deepcopy(decap_maps[best_index])
                best_map_z = copy.deepcopy(decap_maps_z[best_index])


        # Prints current results
        print('Current Best Score:', best_score, 'Associated Map:', best_map)
        print('Generation', Population.current_gen, 'Done!\n')
        Population.increment_generation()

    print('All', Population.generations,' Generations Finished!')
    print('Best Scoring Solution', best_map, 'with score', best_score)

    plt.loglog(opt.freq, np.abs(best_map_z))
    plt.loglog(opt.freq, opt.ztarget)
    plt.xlabel('Frequency in Hertz')
    plt.ylabel('Impedance in Ohms')
    plt.title('Basic GA')
    plt.grid(which='both')
    plt.legend(['Best Scoring Sol', 'ZTarget'])
    print("Time to run in seconds:", time.time() - start_time)
    plt.show()

