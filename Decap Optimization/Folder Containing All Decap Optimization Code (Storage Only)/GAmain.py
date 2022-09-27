# coding: utf-8
# Author: Ling Zhang
# Email : lzd76@mst.edu

import PopInit
import pdn
from config2 import Config
import GenFlow
import numpy as np
import datetime
import random
import os.path
import collections
import copy

def OptionsInit():
    opt = Config()
    return opt


# 8/22 THE STAGNATE COUNTER RESET IS KINDA BROKEN BUT NOT TOO BAD





'''
def create_board(testPop,opt):
    # I don't know if this is even needed but i'll keep it here
    # Get board and settings based on asus specifications
    asus_brd = rf.Network('pdn/ASUS-MST_BrdwithCap_mergeIC_port31VRM.s15p')   # Target file with ports to make board
    new_freq = rf.frequency.Frequency(start=0.01, stop=1000, npoints=601, unit='mhz', sweep_type='log')
    asus_brd = asus_brd.interpolate(new_freq)
    return asus_brd
'''

'''
def encode_2_decode_map(testPop, opt):
    initial_map = testPop.chromosomes.numpy() # An array of initial decap values
    idecap_map = [[0] * opt.total_ports for i in range(len(testPop.chromosomes))]
    for i in range(len(initial_map[:, 0])):
        for j in range(len(initial_map[0, :])):
            idecap_map[i][j] = (pdn.select_decap(initial_map[i, j], opt))  # creates a decapmap for initial population
    return initial_map, idecap_map
'''

def get_initial_maps(testPop, opt):
    decap_map = testPop.chromosomes # An array of initial decap values
    decode_map = [pdn.select_decap(i, opt) for i in range(testPop.numDecaps)] # list of shorted capacitors
    # i might be wasting the testPop.chromosomes variable
    return decap_map, decode_map


if __name__ == '__main__':
# ---------------------------------    Intialization Steps   ------------------------------
    print('Hello There')
    timeStart = datetime.datetime.now()
    opt = OptionsInit()   # Set some settings
    testPop = PopInit.Population(opt) # generate a population's chromosomes
    testPop.create_population(opt)
    initial_map, decode_map = get_initial_maps(testPop, opt)
    # get the initial decap layouts, and a map to match encoded 0 - 10 values with a capacitor to connect

    decap_map = copy.deepcopy(initial_map)
    # just to help me differentiate. Decap map will be constantly overwritten for each gen

    #decap_map[random.randrange(len(decap_map))] = [1,1,0,0,0,2,1,8,2,0,8,4,0,1]
    # thi was for seeding

    #best_score = best_gen_score = opt.fail_score
    best_score = 0
    prev_score = 0
    best_net = [0] * opt.decap_ports
    # set initial holders to hold best scoring network

    unique_best_list = []
    unique_best_master = []     # master holder to not lose solutions
    prev_unique_best_list = []  # holder for previous best solutions for past decap number

    #improve_target_file = 'FancyCurveI2.txt'
    #score_target_file = 'FancyCurveS2.txt'
    #array_file = 'FancyCurveA2.txt'
    #score_dist_file = 'FancyCurveD2.txt'

    #list comprehension might be a change I want to shorten this. May want to rename file path string
    #if os.path.exists(improve_target_file) or os.path.exists(score_target_file) or os.path.exists(array_file)\
     #       or os.path.exists(score_dist_file):
    #    raise FileExistsError('DONT OVERWRITE FILES AGAIN')
    #testfile = open(improve_target_file, 'w')
    #scorefile = open(score_target_file, 'w')
    #arrayfile = open(array_file,'w')
    #distfile = open(score_dist_file,'w')
    #distfile.write('Fail score is {}\n'.format(opt.fail_score))


# ---------------------------------------------------------------------------------
# Begin loop to cover every generation
    while testPop.current_gen <= testPop.generations:

        # Calculate Impedance for initial population and at start of every generation
        # Occurs after elitism is implemented in the previous generation

        solu_net, zpdn = pdn.calc_impedance(decap_map, decode_map, opt)
        # solu_net will return a fully connected network object for each decap layout in decap_map.
        # zpdn will be a list containing the largest impedance for each network object in the specified freq region

        solu_scores = GenFlow.calc_score(solu_net, decap_map, zpdn, opt)
        # calculate the scores of the layouts in decap_map
        tuple_scores = tuple(solu_scores)

        best_gen_score = max(solu_scores)
        best_score = best_gen_score
        # Gets the best scoring solution of in the generation


        score_distribution = collections.Counter(tuple_scores)
        # converts list of scores to a tuple so that I can use Counter to see score distribution

        #print('Score Distribution =', score_distribution)
        #distfile.write('Current Gen = {0}, Score Distribution = {1} (Pre Genetic Operators, '
        #               'Result of crossover from last gen)\n'.format(testPop.current_gen, score_distribution))
        #writes distribution to file

        #best_indices = [indices for indices, ele in enumerate(solu_scores) if int(ele) == int(best_gen_score)]
        best_indices = [indices for indices, ele in enumerate(solu_scores) if abs(ele - best_gen_score) <= .1]
        # Different methods for choosing best indices
        # Using int lets you get away with intermediate scores

        unique_best_list = [decap_map[i] for i in best_indices]

        random_best_index = best_indices[random.randrange(len(best_indices))]
        # get a random index of one of the best scoring solutions
        best_random_net = decap_map[random_best_index]
        # gets a random, but best scoring solution in the population.

        # might not need both the best_gen_net or the beset_random_net tho. One is arguably enough
        best_gen_net = decap_map[solu_scores.index(best_gen_score)]
        # get the first, best scoring solution in the population

        # shouldn't need unique_best
        #unique_best = set(tuple(ele) for index, ele in enumerate(decap_map)
        #                 if (int(best_gen_score)) == (solu_scores[index]))
        # set of all unique solutions having the best score. Each solution should appear only once.


        # list of all unique solutions with best scores. Each solution should only appear once

        if int(best_gen_score) > opt.fail_score:
            for i in unique_best_list:
                if list(i) not in unique_best_master:
                    unique_best_master.append(list(i))


        if best_gen_score <= opt.fail_score:
            decap_num = 'N/A'
        else:
            decap_num = np.count_nonzero(best_random_net)
        print('The best score for this population is:', best_gen_score,
              '\nBest scoring (but random) decap map:', best_random_net, 'Number of Decaps =', decap_num, 'Impedance =', zpdn[random_best_index])


        score_gen = ('Best Score =', best_gen_score, ' For Generation =', testPop.current_gen, 'Decap Map:',
                     best_gen_net)
        #scorefile.write(str(score_gen))
        #scorefile.write('\n')
        num_best_sol = solu_scores.count(max(solu_scores))      # number of solutions with best score
        unique_sol_num = len(unique_best_list)                           # number of unique sol



        #print('Unique Best Solutions:', unique_best_list)
        #scorefile.write(('Number of solutions with best score = {0},'
        #                'Number of unique solutions = {1}, Number of decaps = {2}, Unique and best score = {3}\n\n'
        #                 .format(num_best_sol, unique_sol, decap_num, len(unique_best))))
        #if int(best_gen_score) != opt.fail_score:
        #    arrayfile.write('Generation Num: {0}, '
        #                    'Best Gen Score: {1}, Unique and Best: {2} Sols\n Unique and Best Score Array Maps: {3}\n\n'
        #                    .format(testPop.current_gen,best_gen_score,len(unique_best),unique_best))

        # Write to file stuff to keep notes, no effect on calculations


        # I can definitely rewrite this function using best_gen_score in place of solu_scores. Keep myself from having
        # to iterate through it.
        best_score, best_net = GenFlow.get_min_decap(best_score, best_net, solu_scores, decap_map, opt)
        print('Best Score Currently:', best_score, 'With layout:', best_net, 'Decap Num:', decap_num)

        if (best_gen_score > best_score):
            string = ('The best score improved to', best_score, 'The net is', best_net,
                      'Generation:', testPop.current_gen, 'Decap Number is', decap_num)
            recordstring = str(string)
            #testfile.write(recordstring)
            #testfile.write('\n')

# -------------------------------------------------------------------------------------------------
        # Genetic Operations Begin Here


        if int(best_gen_score) == int(prev_score):
             testPop.increment_stagnate_counter()
             print('Score did not significantly improve. Stagnation counter:', testPop.stagnateCounter)
        else:
             testPop.reset_stagnate_counter()
             print('Score significantly improved: Stagnation counter reset')

        selected_parents = GenFlow.roulette(solu_scores, decap_map, opt)
        # Selected who will reproduce based on roulette (fitness proportional selection)

        crossover_pop = GenFlow.cross_over(selected_parents,testPop,opt)
        #crossover_pop = GenFlow.two_point_cross(selected_parents, testPop, opt)
        #crossover_pop =  GenFlow.uniform_crossover(selected_parents, testPop, opt)
        # Perform crossover on selected parents to create children. These children will be in the next generation, and
        # replace a parents

        mutated_pop = GenFlow.mutation(testPop,crossover_pop,opt, added_zeros = 1)
        # perform mutation on the crossover'ed population

        # calculate scores after crossover and mutation to check for improvements.
        solu_net, zpdn = pdn.calc_impedance(mutated_pop, decode_map, opt)
        solu_scores = GenFlow.calc_score(solu_net, mutated_pop, zpdn, opt)

        best_cross_score = max(solu_scores)
        print('Best score after crossover and mutation is:', best_cross_score)
        # get best score after cross over and mutation.
        # best_cross_score can get worse

        #best_indices = [indices for indices, ele in enumerate(solu_scores) if int(ele) == int(best_cross_score)]
        best_indices = [indices for indices, ele in enumerate(solu_scores) if abs(ele - best_cross_score) <= .1]
        # Different methods for choosing best indices
        # Using int lets you get away with intermediate scores

        if int(best_cross_score) > int(best_gen_score):
            #testPop.reset_stagnate_counter()
            #print('Score Significantly Improved, stagnation counter reset')
            #prev_unique_best_list = copy.deepcopy(unique_best_list)
            prev_unique_best_list = copy.deepcopy(unique_best_master)

            unique_best_master = [list(mutated_pop[i]) for i in best_indices]
            #unique_best_master = [list(mutated_pop[i]) for i in best_indices if abs(best_cross_score - solu_scores[i]) <= .1]
            # For some reason saying  unique_best_master = [ (mutated_pop[i]) for i in best_indices ]
            # unique_best_master turns into a list of arrays instead of a list of lists. looks stupid so probably
            # im missing something

            #print(unique_best_master)
            print('List of Unique Solutions for Previous Best Decap Number Updated')
            decap_map = copy.deepcopy(mutated_pop)
            # Generate a new unique list for elitism

        if int(best_cross_score) >= int(best_gen_score):
            unique_best = set(tuple(ele) for index, ele in enumerate(mutated_pop)
                              if int(solu_scores[index]) == int(best_cross_score) and solu_scores[index] > opt.fail_score)
            unique_best_list = [list(i) for i in unique_best]
            if best_cross_score != opt.fail_score:
                for i in unique_best_list:
                    if i not in unique_best_master:
                        unique_best_master.append(i)
                        print('Appended. Master list updated')





            # Update unique best list after crossover and mutation if score doesn't get worse
            # If score gets worse, don't want to get a worse unique best list

        #print(set(tuple(ele) for index, ele in enumerate(mutated_pop)
         #                     if solu_scores[index] == best_cross_score), 'set')
        #print(list(set(tuple(ele) for index, ele in enumerate(mutated_pop)
          #                    if solu_scores[index] == best_cross_score)), 'list')

        if testPop.stagnateCounter == opt.min_stagnation_gens:
             #scorefile.write('Gene Shakeup Occured. At Generation = {}\n\n'.format(testPop.current_gen))
             if best_gen_score == opt.fail_score and best_cross_score == opt.fail_score:
                decap_map = GenFlow.gene_shakeup(mutated_pop, best_indices, opt, best_score, zero_bias = 0)
                print('No solutions meeting target impedance found, creating all new chromosomes')
             else:
                # might want to also consider the order of this
                #decap_map = GenFlow.zeros_diversity(decap_map, best_score, opt, testPop)

                decap_map = GenFlow.insert_prev_score_sols(mutated_pop, prev_unique_best_list, best_gen_score, opt)
                # The insert_prev_score_sols is useless if I start off with a super good sol.
                # If I seed initial population with a N decap solution, then I have no N + 1 or
                # N + 2 solutions to insert. And if a N - 1 decap solution isn't found, then
                # this function does nothing completely

                # this probably works better with elitism2() then elitism1()
                print('Population has stagnated, inserting all unique solutions associated with previous best score')
             testPop.reset_stagnate_counter()
             print('Stagnation counter reset.')


        #decap_map = GenFlow.elitism(best_gen_score, best_random_net, best_score, best_net, mutated_pop, opt)
        #decap_map = GenFlow.elitism2(unique_best_list, mutated_pop, best_gen_score, opt)

        if int(best_cross_score) <= int(best_gen_score):
            # do elitism only if # of decaps doesn't improve from crossover and/or mutation
            # This way, if the decap num improves through crossover/mutation, I won't accidently overwrite it with elitism,
            # and not before they have a chance to be recorded in unique lists.

            # decap_map = GenFlow.elitism2(unique_best_master, mutated_pop, best_cross_score, opt)
            # decap_map = GenFlow.elitism2(unique_best_master, mutated_pop, best_cross_score, opt)
            decap_map = GenFlow.elitism3(unique_best_master, mutated_pop, best_gen_score, opt, replace_limit = 5)
            #print('Elitism Occured')
        else:
            print('Score significantly improved, elitism will not take place for this generation')

        # After roulette selection, crossover, mutated_pop, elitism, decap_map will take on the decap layout
        # of the next generation

        print('Generation', testPop.current_gen, 'completed\n')
        testPop.current_gen += 1

        prev_score = best_gen_score

        #print('Prev list =', prev_unique_best_list)
        #print('Master =', unique_best_master)

#       #Loops to next generation
# ------------------------------------------------------------------------------------------

    timeTaken = ('Start time:', timeStart, 'End Time', datetime.datetime.now())
    #testfile.write(str(timeTaken))
    #testfile.close()
    #scorefile.close()
    #arrayfile.close()
    #distfile.close()
    connected_net,best_z = pdn.calc_impedance([best_net], decode_map, opt)
    print('Impedance of best layout is', best_z)
    pdn.testPlot(connected_net,opt)






