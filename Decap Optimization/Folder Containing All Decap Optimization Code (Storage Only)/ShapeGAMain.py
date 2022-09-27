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
    #initial_z = np.absolute(opt.input_net.z[:,0,0]) # Get Impedance of Board
    initial_z = opt.input_net.z

    initial_sol_found = False
    initial_sol_index = []
    final_maps = []
    final_maps_scores = []
    best_map = [0]                                  # Variable to store best scoring sol
    best_score = 0                                  # Variable to store best score
    best_z = []
    min_zero_map = []                               # Variable to store minimum decap solution
    min_zero_index = []
    post_decap_maps_scores = []

    R = .01
    Zmax = .02
    #opt.ztarget = get_target_z_RL(R, Zmax, fstart=1e4, fstop=20e6, nf=201, interp='log')



    ###### Pre Processing For Any Functions #######
    shift_f = pdn1.target_shift(opt)
    print('Frequency shift point set as', shift_f)
    #min_z_f = pdn1.find_minimum_freq_pt(opt, shift_f)
    #print('Ideal minimum z frequency set as', min_z_f)



    # calculate distance (Lloop) roughly
    distances = pdn1.calculate_rough_distance(opt, cap_objs_z)

    #test = [6,0,0,0,0,0,0,0]
    #pdn1.calc_first_antires(test, cap_objs_z, opt)

    ##### Determining 'bulk' and best local min capacitor #####
    bulk_z_index, bulk_cap = pdn1.find_bulk_cap(shift_f, cap_objs_z, initial_z, opt)
    #print('The bulk cap is:', bulk_cap)

    # Get the smallest capacitor needed
    min_z_index, min_cap = pdn1.find_min_cap(shift_f, cap_objs_z, initial_z, opt)
    #min_z_index, min_cap = pdn1.find_min_cap2(bulk_cap, shift_f, cap_objs_z, initial_z, opt)
    #min_cap = 2
    #bulk_cap = 10
    print('Bulk capacitor is decap', bulk_cap, '\nCapacitor to produce best final local min is decap', min_cap)
    tester = [[0] * opt.num_decaps]
    tester[0] = bulk_cap

    tester_z = pdn1.new_connect_n_decap(opt.input_net.z,tester,cap_objs_z,opt)

    mins = pdn1.find_mins(tester_z,local_only=False)
    print(mins)


    min_z_f = opt.freq[mins[0]]

    print('Ideal minimum z frequency set as', min_z_f)

    bulk_full = [bulk_cap] * opt.decap_ports
    bulk_full_z = pdn1.new_connect_n_decap(opt.input_net.z, bulk_full, cap_objs_z, opt)
    min_full = [min_cap] * opt.decap_ports
    min_full_z = pdn1.new_connect_n_decap(opt.input_net.z, min_full, cap_objs_z, opt)

    ########## Manipulation of Initial Population #########
    # for i in range(len(decap_maps)):   # i don't know why doing just in in decap_maps isn't working
    #     decap_maps[i] = GA1.add_number(decap_maps[i], 9, opt, total_added=math.floor(opt.decap_ports/4))
    #for i in range(len(decap_maps)):  # i don't know why doing just in in decap_maps isn't working
    #     decap_maps[i] = GA1.add_number(decap_maps[i], 1, opt, total_added=math.floor(opt.decap_ports / 4))
    #print('Initial Map After Seeding', decap_maps)

    ##### Forcibly allow only caps smaller than and equal to the chosen bulk capacitor
    ## i know this can be written better, ill do it later when i get the syntax correct
    # also easier if i just don't let those bigger caps be produced when you create the population
    # but this is less commital
    for i in decap_maps:
        for ind, cap in enumerate(i):
            i[ind] = cap if cap <= bulk_cap else random.randrange(0,bulk_cap+1)

    ####### Begin GA ######
    while Population.current_gen <= Population.generations:

        print('Generation', Population.current_gen, 'Start!')

        # Calculate z
        decap_maps_z = [pdn1.new_connect_n_decap(opt.input_net.z,decap_maps[i],cap_objs_z,opt) for i in range(Population.population)]


        # Calculate scores and look for solutions in initial population

        if Population.current_gen == Population.startGen:
            # Check to see if an initial solution exists in initial population
            initial_sol_found, initial_sol_index = GA1.check_for_sol(decap_maps_z,opt)
            if initial_sol_found is True:
                best_map = copy.deepcopy(decap_maps[initial_sol_index])
                min_zero_map = copy.deepcopy(best_map)
                print('Initial Solution Found In Initial Population. Initial Solution:', min_zero_map)

        if not initial_sol_found:
            # Calculate scores based on finding an initial solution
            #decap_maps_scores = GA1.initial_sol_score(decap_maps_z, opt)
            decap_maps_scores = GA1.initial_sol_score2(decap_maps, decap_maps_z, opt, min_z_f, shift_f, bulk_full_z, min_full_z, cap_objs_z)
            print('Start of Gen:,', Population.current_gen, 'Score while looking for initial sol:', decap_maps_scores)

        else:
            # calculate scores based on finding a minimum decap solution
            decap_maps_scores, min_zero_map = GA1.calc_score(decap_maps, decap_maps_z,
                                                             opt, min_zero_map, min_z_f, bulk_full_z, min_full_z,
                                                             distances, shift_f)
            print('Scores at the start of Gen:', Population.current_gen,':', decap_maps_scores)

        if Population.current_gen == Population.startGen:
            best_score = max(decap_maps_scores)
            best_index = decap_maps_scores.index(best_score)
            best_map = copy.deepcopy(decap_maps[best_index])
            best_map_z = copy.deepcopy(decap_maps_z[best_index])

        ########## Begin GA operations ########
        # Select parents for crossover based on score
        selected_parents = GA1.roulette(decap_maps_scores, decap_maps)

        # THIS NEEDS TO BE EDITED I THINK
        # the crossover2 function is kind of broken
        if initial_sol_found:
            # Crossover based on minimizing # of capacitors
            cross_population = GA1.cross_over(selected_parents, Population, opt)
            # perform random crossover of best scoring population with lower scoring
        else:
            # Crossover to find initial solution.
            cross_population = GA1.cross_over(selected_parents, Population, opt)

        # Perform Mutation
        #mutated_population = GA1.mutation(Population, cross_population, opt, initial_sol_found, added_zeros=1) # perform mutation
        mutated_population = GA1.mutation2(Population, cross_population, opt)


        # Get population after performing GA operators
        post_decap_maps = copy.deepcopy(mutated_population)

        ##### Forcibly allow only caps smaller than and equal to the chosen bulk capacitor
        ## i know this can be written better, ill do it later when i get the syntax correct
        # also i can just not let it mutate to those values rather than doing it twice
        for i in post_decap_maps:
            for ind, cap in enumerate(i):
                i[ind] = cap if cap <= bulk_cap else random.randrange(1,bulk_cap+1)

        # Twin Removal, remove and regenerate duplicate solutions
        #post_decap_maps = GA1.twin_removal(post_decap_maps,opt)

        # Calculate the impedance of your new solution set
        post_decap_maps_z = [pdn1.new_connect_n_decap(opt.input_net.z, post_decap_maps[i], cap_objs_z, opt) for i in
                             range(Population.population)]

        for i in post_decap_maps:
            for ind, cap in enumerate(i):
                i[ind] = cap if cap <= bulk_cap else random.randrange(1, bulk_cap + 1)


        # Check for solutions and assign scores based on Population post GA Ops, and whether initial sol found
        if not initial_sol_found:
            initial_sol_found, initial_sol_index = GA1.check_for_sol(post_decap_maps_z, opt)

            if initial_sol_found is True:
                best_map = copy.deepcopy(post_decap_maps[initial_sol_index])
                min_zero_map = copy.deepcopy(best_map)
                print('Initial Solution Found. Solution:', best_map)
                print('Recreating Population with Initial Solution Inserted')

                # Recreate Population After Finding Initial Solution.
                Population.create_population(opt)
                # Reinsert Initial Solution
                Population.chromosomes[0] = copy.deepcopy(best_map) # insert the initial solution in there
                best_score = 0  # Reset scoring so you can go to the second scoring method
                post_decap_maps_scores, min_zero_map = GA1.calc_score(post_decap_maps, post_decap_maps_z,
                                                                      opt, min_zero_map, min_z_f,  bulk_full_z, min_full_z,
                                                                      distances, shift_f)

            if initial_sol_found is False:
                # Do scoring method to find initial solution
                #post_decap_maps_scores = GA1.initial_sol_score(post_decap_maps_z, opt)
                post_decap_maps_scores = GA1.initial_sol_score2(post_decap_maps, post_decap_maps_z, opt, min_z_f, shift_f, bulk_full_z, min_full_z, cap_objs_z)
                print('After GA Ops of Gen:', Population.current_gen, ', Scoring while looking for initial sol:', post_decap_maps_scores)
        else:
            # Use scoring based on minimizing decap #
            post_decap_maps_scores, min_zero_map = GA1.calc_score(post_decap_maps, post_decap_maps_z,
                                                                  opt,  min_zero_map, min_z_f,  bulk_full_z, min_full_z,
                                                                  distances, shift_f)

        # Update best scoring solution
        if (max(post_decap_maps_scores) > best_score):
            print('Score Improved. Current Scores:', post_decap_maps_scores)
            best_score = max(post_decap_maps_scores)
            best_index = post_decap_maps_scores.index(best_score)
            best_map = copy.deepcopy(post_decap_maps[best_index])
            best_map_z = copy.deepcopy(post_decap_maps_z[best_index])


        # Perform elitism
        # also commented elitism out 9/16
        post_decap_maps = GA1.elitism(best_score, best_map, post_decap_maps, post_decap_maps_scores, initial_sol_found,
                                      min_zero_map)

        print('Scores after genetic operator', post_decap_maps_scores)
        # if not initial_sol_found:
        #
        #
        #     # Do elitism to look for initial solution
        #     post_decap_maps = GA1.elitism(best_score, best_map, post_decap_maps,  post_decap_maps_scores,initial_sol_found,min_zero_map)
        #
        # if initial_sol_found:
        #     # Do elitism method after initial solution has been found
        #     post_decap_maps = GA1.elitism(best_score, best_map, post_decap_maps, post_decap_maps_scores,initial_sol_found,min_zero_map)

        decap_maps = copy.deepcopy(post_decap_maps)

        print('Genetic Operators Complete')

        # Store Maps For Final Processing if it is the final generation
        if Population.current_gen == Population.generations:
            final_maps = copy.deepcopy(post_decap_maps)
            final_maps_scores = copy.deepcopy(post_decap_maps_scores)


        # Prints current results
        print('Current Best Score:', best_score, 'Associated Map:', best_map)
        #print('Max z of Best Scoring Map =', max(np.absolute(best_map_z)))
        print('Map with minimum decap number =', min_zero_map)
        print('Generation', Population.current_gen, 'Done!\n')
        Population.increment_generation()

    print('All', Population.generations,' Generations Finished!')

    #### Begin Post Processing #####

    best_z = []
    min_decap_z = []

    #### Post Processing Based On If Any Solution Satisfying Target Was Found #####

    # If Solution Satisfying Target was Found
    if initial_sol_found is True:

        # Check if the min decap and best map sol can have their decap # decreased (Primitively)
        min_zero_map = pdn1.final_check(min_zero_map, opt, cap_objs_z)  # check if min decap sol can be improved
        best_map = pdn1.final_check(best_map,opt,cap_objs_z)            # check if best scoring sol can be improved

        # Recalculate Impedances, whether it improves or not
        min_decap_z = [pdn1.new_connect_n_decap(opt.input_net.z, min_zero_map, cap_objs_z, opt)]
        best_z = [pdn1.new_connect_n_decap(opt.input_net.z, best_map, cap_objs_z, opt)]

        # Print Final Results
        print('Minimal Decap Map is', min_zero_map, 'with', np.count_nonzero(min_zero_map), 'capacitors')
        print('Best Scoring Map is', best_map, 'with', np.count_nonzero(best_map), 'capacitors')       # print best scoring map
        print('Due to the final check, best scoring map may have less decaps!\n')

        # Check to see how many sols have more zeros than min zero map  (Troubleshooting for me)
        num_empty = opt.decap_ports - np.count_nonzero(min_zero_map)
        check = 0
        for i in final_maps:
            if opt.decap_ports - np.count_nonzero(i) > num_empty:
                check = check + 1
        print('Number of maps with more zeros than the min decap sol is:', check)

    # Post processing if no solution was found satisfying the target
    else:

        # Calculate impedance of 'solution' closest to satisfying the target
        best_z = [pdn1.new_connect_n_decap(opt.input_net.z, best_map, cap_objs_z, opt)]

        # if no initial solution found, set min decap map z to that of the closest solution
        min_decap_z = copy.deepcopy(best_z)
        min_zero_map = best_map

        # Print final results
        print('No solution found satisfying target impedance')
        print('Closest solution found is:', best_map)
        print('Min Decap Map is set the same:', min_zero_map)

    plt.loglog(opt.freq, opt.ztarget)
    plt.loglog(opt.freq, np.abs(min_decap_z[0]))
    plt.loglog(opt.freq, np.abs(best_z[0]))

    title = "Frequency vs Impedance"
    plt.title(title)
    plt.legend(['Ztarget', 'Minimum Capacitor Solution','Best Scoring Solution'])
    plt.grid(which='both')
    print("Time to run in seconds:", time.time() - start_time)
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Impedance |Z| in Ohms')
    plt.show()



