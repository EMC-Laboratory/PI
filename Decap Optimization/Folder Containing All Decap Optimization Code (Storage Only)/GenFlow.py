
import numpy as np
import pdn
import time
import random
import copy
import math



def calc_score(solu_net, decap_map, zpdn, opt):
    # decap_map = the list of encoded numbers
    solu_scores = []
    # Port prio: 10, 7, 12, 9, 3, 13, 8, 15, 2, 6, 14, 4, 11, 5
    # prio_list = [8,5,10,7,1,11,6,13]  #based on inductance calcs, 7/29, for 8 decaps, these ports must have decaps
    # zeros_list = [0,2,3,4,9,12]
    #for i in range(len(solu_net)):  # each solution set
    #    score_mod = 0
    #    if np.count_nonzero(np.greater(zpdn[i], opt.ztarget)) == 0:  #if you meet impedance target
        #     #[1, 1, 0, 0, 0, 2, 1, 8, 2, 0, 8, 4, 0, 1]
        #     for j in zeros_list:
        #         if decap_map[i][j] == 0:
        #             score_mod += 1
        #     solu_scores.append(opt.no_empty_ports_score + score_mod)
        #     #+ len(decap_map[i]) - np.count_nonzero(decap_map[i]))
        #     scoring based on port priority

    #        (solu_scores.append(opt.no_empty_ports_score) if np.count_nonzero(decap_map[i]) == len(decap_map[i])
    #        else solu_scores.append(opt.no_empty_ports_score + (len(decap_map[i])- np.count_nonzero(decap_map[i]))))
            #scoring based entirely on random chance


        # if meet target impedance but has no empty ports, give no_empty_ports_score
        # if meet target impedance and has empty ports, give no_empty_ports_score + number of empty ports


     #   else:
     #       solu_scores.append(opt.fail_score)
        #     #if impedance target of a network is not met, give bad fitness score
        #solu_scores[-1] = solu_scores[-1] * (1 + 100 * abs((max(opt.ztarget) - zpdn[i]))
        # trying out effect of adding a modifier based on how low the max impedance is

    # trying the z-mask paper method

    # SIDE NOTE FOR ME, WHEN INTEGRATING THE FASTER CODE USING NETWORK OBJECT< USE map.z for easier time
    for i in range(len(solu_net)):
        pts_below_z = [pt for pt in range(opt.nf-180) if (abs(solu_net[i].z[opt.nf-pt-1,opt.ic_port-1,opt.ic_port-1]) <= opt.ztarget[opt.nf-pt-1])]
        if len(pts_below_z) == opt.nf-180:  # if meet impedance requirements, givesmall boost to score
            solu_scores.append((opt.multiplier * len(pts_below_z)) + (len(decap_map[i])- np.count_nonzero(decap_map[i])+1))
        else:
            solu_scores.append((opt.multiplier * len(pts_below_z)))
    return solu_scores  # returns scores for all members in a generation





def get_min_decap(prev_score, prev_map, solu_scores, decap_map, opt):
    # This function will just hold the layout with the minimum number of decaps that meet impedance requirements
    # throughout generations.

    best_index = solu_scores.index(max(solu_scores))
    #min_impedance_index = zpdn.index(min([zpdn[i] for i in best_indices]))
    base_score = opt.nf * opt.multiplier  # use this variable for comparison purposes

    #if solu_scores[best_index] <= base_score:
    #    print('Impedance requirements not met in this generation')
     #   return prev_score, prev_map
    # best_index should give the first instance of the best scoring member of the population
    if solu_scores[best_index] > prev_score:
        print('Best score and layout updated')
        return solu_scores[best_index], decap_map[best_index]
    else:
        print('No better score, previous best score and layout retained')
        return prev_score, prev_map

def min_impedance_map(zpdn, decap_map, best_indices):
    pass







def roulette(solu_scores, decap_map, opt):
    # Use roulette to generate new population
    # Roulette might take too long
    selected_parents = copy.deepcopy(decap_map)
    indiv_fit = solu_scores.copy() # don't need a deep copy because solu_scores should be 1 layer deep

    #indiv_fit = np.zeros(len(solu_scores))
    #for x,y in enumerate(solu_scores):
    #    indiv_fit[x] = 1/y
    #for i in range(len((solu_scores))):
        #indiv_fit[i] = 1/solu_scores[i]   # won't  have to worry about divide by zero as 0 decaps will never work

    total_fit = sum(indiv_fit)
    fit_percent = [indiv_fit[i]/total_fit for i in range(len(indiv_fit))]
    roulette = np.cumsum(fit_percent)
    for i in range(len(decap_map)):
        # Create new population
        rand_event = random.random()
        for j in range(len(roulette)):
            if rand_event < roulette[j]: # roulette wheel to calculate next generation
                selected_parents[i] = decap_map[j]
                break
    return selected_parents


#For crossover and two point crossover, might be replacing wrong parents. It was fine befoer shuffle, not fine after

def cross_over(decap_map, popu, opt):
    # decap_map techinically = parents map layout gotten from the selection method.
    # popu = population class object
    # Using new generation created by roulette wheel, do crossover to create new chromosomes
    cross_pop = copy.deepcopy(decap_map)  # Holder for cross over population
    #need deepcopy here, I want to modify values in cross_pop without changing values in decap_map

    cross_parents = [x for x, _ in enumerate(decap_map) if random.random() < popu.crossoverRate]
    # get indices for parents that will crossover

    random.shuffle(cross_parents)
    # Since theoretically, by roulette roll, most parents will be similar and possibly occur in series
    # I want to shuffle it to see if i can encourage more interesting crossovers.

    if len(cross_parents) >= 2:  # Two parents at least must exist for crossover to be done
        for i, parent_pos in enumerate(cross_parents):
            # can prob simplify, visually at least, with enumerate() later
            point = random.randrange(1, opt.decap_ports-1)  # Cross point designation
            left_parent = random.randrange(1,3)    # 1 = i'th parent is on left. 2 = i+1 parent in on left
            if i + 1 == len(cross_parents):
                if left_parent == 1:
                    new_chromo = [decap_map[parent_pos][j] for j in range(point)]
                    new_chromo = new_chromo + [decap_map[cross_parents[0]][j] for j in range(point, opt.decap_ports)]
                    cross_pop[parent_pos] = new_chromo
                    # don't need the extra assignment, but I think it looks better if i don't go to next line
                else: # the i + 1 parent is on the left
                    new_chromo = [decap_map[cross_parents[0]][j] for j in range(point, opt.decap_ports)]
                    new_chromo = new_chromo + [decap_map[parent_pos][j] for j in range(point)]
                    cross_pop[parent_pos] = new_chromo
            else:
                if left_parent == 1:
                    new_chromo = [decap_map[parent_pos][j] for j in range(point)]
                    new_chromo = new_chromo + [decap_map[cross_parents[i + 1]][j] for j in range(point, opt.decap_ports)]
                    cross_pop[parent_pos] = new_chromo
                else:
                    new_chromo = [decap_map[cross_parents[i+1]][j] for j in range(point, opt.decap_ports)]
                    new_chromo = new_chromo + [decap_map[parent_pos][j] for j in range(point)]
                    cross_pop[parent_pos] = new_chromo
    return cross_pop

def two_point_cross(decap_map, popu, opt):
    cross_pop = copy.deepcopy(decap_map)
    cross_parents = [x for x, _ in enumerate(decap_map) if random.random() < popu.crossoverRate]
    random.shuffle(cross_parents)
    if len(cross_parents) >= 2:
        for i,parent_pos in enumerate(cross_parents):
            left_parent = random.randrange(0, 2) # pick left parent
            # Parents = [i, i+1], if left parent, take genes starting with i. If right parent, start with i + 1 genes
            P1 = random.randrange(1, opt.decap_ports - 1) # cross point 1
            P2 = random.randrange(1, opt.decap_ports - 1) # cross point 2
            while (P2 == P1):
                P2 = random.randrange(1, opt.decap_ports)
            leftP = min(P1,P2)
            rightP = max(P1,P2)

            if i + 1 == len(cross_parents):
                if left_parent:
                    new_chromo = [decap_map[parent_pos][j] for j in range(leftP)]
                    new_chromo = new_chromo + [decap_map[cross_parents[0]][j] for j in range(leftP, rightP)]
                    cross_pop[parent_pos] = (new_chromo +
                                             [decap_map[parent_pos][j] for j in range(rightP, opt.decap_ports)])
                else:
                    new_chromo = [decap_map[cross_parents[0]][j] for j in range(leftP)]
                    new_chromo = new_chromo + [decap_map[parent_pos][j] for j in range(leftP, rightP)]
                    cross_pop[parent_pos] = (new_chromo +
                                               [decap_map[cross_parents[0]][j] for j in range(rightP, opt.decap_ports)])
            else:
                if left_parent:
                    new_chromo = [decap_map[parent_pos][j] for j in range(leftP)]
                    new_chromo = new_chromo + [decap_map[cross_parents[i + 1]][j] for j in range(leftP, rightP)]
                    cross_pop[parent_pos] = (new_chromo +
                                               [decap_map[parent_pos][j] for j in range(rightP, opt.decap_ports)])
                else:
                    new_chromo = [decap_map[cross_parents[i + 1]][j] for j in range(leftP)]
                    new_chromo = new_chromo + [decap_map[parent_pos][j] for j in range(leftP, rightP)]
                    cross_pop[parent_pos] = (new_chromo  +
                                             [decap_map[cross_parents[i+1]][j] for j in range(rightP, opt.decap_ports)])
    return cross_pop

def uniform_crossover(decap_map, popu, opt):
    cross_pop = copy.deepcopy(decap_map)
    cross_parents = [x for x, _ in enumerate(decap_map) if random.random() < popu.crossoverRate]
    random.shuffle(cross_parents)

    if len(cross_parents) >= 2:
        for index, parent_pos in enumerate(cross_parents):
            # index = indices of cross_parents. Parent_pos = parent positions in decap_map
            gene_distri = [random.randrange(2) for _ in range(opt.decap_ports)]
            # 0 will represent the i'th parent, 1 will represent the i + 1 parent. values in gene_distri will tell you
            # what genes to take from each parent

            if index + 1 == len(cross_parents): # if on the last index of cross_parents
                for gene_pos, parent in enumerate(gene_distri):  # for each gene, and the parent you take it from
                    cross_pop[parent_pos][gene_pos] =  (decap_map[parent_pos][gene_pos] if not parent else
                                                         decap_map[cross_parents[0]][gene_pos])
                    #cross_pop[chromo being overwriteen][gene being overwritten]
            else:
                for gene_pos,parent in enumerate(gene_distri):  # for each gene, and the parent you take it from
                    cross_pop[parent_pos][gene_pos] =  (decap_map[parent_pos][gene_pos] if not parent else
                                                         decap_map[cross_parents[index+1]][gene_pos])
    return cross_pop


def mutation(popu, cross_pop, opt, added_zeros = 1):
    # I can definitely simplify this entire function into a form like gene_shakeup. But not a huge priority

    num_mut = int(popu.mutationRate * opt.decap_ports * popu.population)
    # lazy way of rounding down to an int for the # of mutations

    mutated_pop = copy.deepcopy(cross_pop)

    if num_mut == 0:
        return mutated_pop  # no mutations occur in this case
    for i in range(num_mut):
        ran_chromosome = random.randrange(len(cross_pop))  # get random chromosome
        ran_gene = random.randrange(opt.decap_ports)  # get a random gene
        rand_decap = random.randrange(popu.numDecaps + added_zeros)
        if rand_decap > (popu.numDecaps - 1):
            rand_decap = 0
        while rand_decap == mutated_pop[ran_chromosome][ran_gene]:
            rand_decap = random.randrange(popu.numDecaps + added_zeros)
            if rand_decap > (popu.numDecaps - 1):
                rand_decap = 0
        mutated_pop[ran_chromosome][ran_gene] = rand_decap
    return mutated_pop
'''
    elif not zero_bias:
        for i in range(1,num_mut+1): # i don't know remember why I have the i range like that but it is
            ran_chromosome = random.randrange(len(cross_pop)) # get random chromosome
            ran_gene = random.randrange(opt.decap_ports) # get a random gene
            rand_decap = random.randrange(popu.numDecaps)
            #mutated_pop[ran_chromosome][ran_gene] = random.randrange(popu.numDecaps)
            #while mutated_pop[ran_chromosome][ran_gene] == random.randrange(popu.numDecaps):
            while rand_decap == mutated_pop[ran_chromosome][ran_gene]:
                rand_decap = random.randrange(popu.numDecaps)
                #mutated_pop[ran_chromosome][ran_gene] = random.randrange(popu.numDecaps)
                # too make sure i actually replace with a new number
            mutated_pop[ran_chromosome][ran_gene] = rand_decap
            #print('Mutation', i, ":", mutated_pop)
            # at a random gene of a random chromosome, generate random decap.
        #print('Mutation Occurred')
        return mutated_pop
    elif zero_bias:
        for i in range(1, num_mut + 1):
            ran_chromosome = random.randrange(len(cross_pop))  # get random chromosome
            ran_gene = random.randrange(opt.decap_ports)  # get a random gene
            rand_decap = random.randrange(popu.numDecaps + added_zeros)
            if rand_decap >= popu.numDecaps:
                rand_decap = 0
            while rand_decap == mutated_pop[ran_chromosome][ran_gene]:
                rand_decap = random.randrange(popu.numDecaps + added_zeros)
                if rand_decap >= popu.numDecaps:  # popu.numDecaps is equal to the number of decaps selections, + 1 for
                                                  # not putting in a decap at all
                    rand_decap = 0
            mutated_pop[ran_chromosome][ran_gene] = rand_decap
    '''
def chromo_shakeup(decap_map):
    new_gene_layout = copy.deepcopy(decap_map)
    for i in new_gene_layout:
        random.shuffle(i)
    return new_gene_layout

def gene_shakeup(decap_map, best_indices, opt, best_score, zero_bias = 0):
    # as it currently is, if the best score is the fail score, the genes won't be shaken up, they'll be kept
    # exactly the same.

    # this works, but is written in stupidly. Will change later

    shaken_map = copy.deepcopy(decap_map)
    if zero_bias < 0:
        print('zero_bias set as less than 0 which is invalid. Setting zero_bias to 0')
        zero_bias = 0
    print('Test:', zero_bias)
    for i in range(len(decap_map)):  # might be able to simplify this
        if best_score != opt.fail_score and i not in best_indices:     # all maps with scores worse than the best score
            shaken_map[i] = np.random.randint(opt.num_decaps + zero_bias, size=(1, opt.decap_ports))
            # shake up gene pool by creating new random chromos for stagnated populations
            # new random chromos will overwrite only those whose score is wirse then the best score
            for j in range(len(shaken_map[i])):
            # can't use for j in shaken_map[i] i dont think because i need the actual index.
                if shaken_map[i][j] > (opt.num_decaps - 1):
                    shaken_map[i][j] = 0
        elif best_score == opt.fail_score: # if stagnate on the fail score, replace everything
            shaken_map[i] = np.random.randint(opt.num_decaps + zero_bias, size=(1, opt.decap_ports))
            # shake up gene pool by creating new random chromos for stagnated populations
            # new random chromos will overwrite only those whose score is less then the best score
            for j in range(len(shaken_map[i])):
            # can't use for j in shaken_map[i] i dont think because i need the actual index.
                if shaken_map[i][j] > (opt.num_decaps - 1):
                    shaken_map[i][j] = 0
    print('Shook up gene pool due to stagnation. Solutions with best scores will be unchanged')
    return shaken_map

def insert_prev_score_sols(decap_map, prev_unique_best_list, prev_score, opt):

    shook_map = copy.deepcopy(decap_map)
    if prev_score == opt.fail_score:
        print('There are no previous, worse scoring solutions, that satisfy target impedance,'
              ' that can be inserted')
        return shook_map
    for i in prev_unique_best_list:
        shook_map[random.randrange(len(shook_map))] = i  # this line will do nothing if prev_unique_best_list is []
    return shook_map


# THIS ONE MAY COME IN HANDY LATER
#
# def zeros_diversity(decap_map, best_score, opt, popu, new_set_num = 5):
#     #This section could be simplified a bit. Would be kinda meaningless to do but can be done
#
#     # I might want to consider instead of shuffling, also changing up the nonzero values
#     # might also want to consider, for num zeros > total zeros, force the number of zeros to go to total_zeros
#     zeroed_map = copy.deepcopy(decap_map)
#
#     if (best_score == opt.fail_score):
#         return zeroed_map
#
#     total_zeros = opt.decap_ports - best_score  # should return number of zeros
#     replace_pos = random.sample(list(range(popu.population)),new_set_num)
#
#     if(new_set_num < 0 or not isinstance(new_set_num, int)):
#         raise ValueError('Negative Values not accepted. An integer >= 0 is required, representing the number of'
#                          'random solutions with randomly placed zeros.')
#     elif new_set_num == 0:
#         return zeroed_map
#     else:
#         print('Diversity Zeros Done')
#         for i in replace_pos:
#             num_zeros = opt.decap_ports - np.count_nonzero(zeroed_map[i])
#             if num_zeros < total_zeros:
#                 non_zero_pos = [j for j, _ in enumerate(zeroed_map[i]) if j != 0]
#                 for k in random.sample(non_zero_pos, total_zeros - num_zeros):
#                     zeroed_map[i][k] = 0
#                     random.shuffle(zeroed_map[i])
#             else:
#                 random.shuffle(zeroed_map[i])
#         # i is the chromosome being modified
#         # j are the potential gene positions of chromosome[i] being modified (potential being non_zero genes)
#         # k are the actual gene positions being replaced with a 0
#     return zeroed_map


def elitism(best_gen_score, best_random_net, best_score, best_net, mutated_pop, opt):
    # Instead of doing elitism every time, lets see what happens if I do elitism only if score gets worse.
    # see what happens
    end_pop = mutated_pop
    # don't think I need a deepycopy because I am done with mutated_pop at this point
    if best_score != opt.fail_score:
        if best_gen_score == best_score:
            randPos = random.randrange(len(mutated_pop))
            end_pop[randPos] = best_net
            randPos2 = random.randrange(len(mutated_pop))
            while randPos2 == randPos:
                randPos2 = random.randrange(len(mutated_pop))
            end_pop[randPos2] = best_random_net
    return end_pop

        # If the best score of a generation is equal to the best score overall, I will insert both the map
        # of the best overall score, and a random map of the best score of the generation, to the next generation
        #

    # if best_net not in end_pop:
    #     print('Hello There')
    #     randPos = random.randrange(len(mutated_pop))
    #     end_pop[randPos] = best_net
    #     if best_random_net not in end_pop:
    #         randPos2 = random.randrange(len(mutated_pop))
    #         while (randPos2 == randPos):
    #             randPos2 = random.randrange(len(mutated_pop))
    #         end_pop[randPos2] = best_random_net

def elitism2(unique_best_list, decap_map, best_gen_score, opt, replace_limit = None):
    end_pop = copy.deepcopy(decap_map)

    if not isinstance(replace_limit, int) and replace_limit is not None:
        raise ValueError('Replace limit must be a positive integer > 0 or Nonetype')
    elif isinstance(replace_limit, int) and replace_limit <= 0:
        raise ValueError('Replace limit must be a positive integer > 0 or Nonetype')
    elif isinstance(replace_limit, int) and replace_limit > len(unique_best_list):
        print('The elitims replacement limit is > than the number of unique best solutions. '
              'No elitism replacement limit will be set to no limit for this generation')
        replace_limit = None


    if best_gen_score != opt.fail_score:
        if replace_limit is None:
            # if its a long list, which it may be, I don'tparticular care if they overwrite each other.
            # prevents saturation
            for i in unique_best_list:
                    randPos = random.randrange(len(end_pop))
                    end_pop[randPos] = i
        else:
            replace_pos = random.sample(list(range(len(decap_map))), replace_limit)
            for index, sol in enumerate(random.sample(unique_best_list, replace_limit)):
                end_pop[replace_pos[index]] = sol

    return end_pop




def elitism3(unique_best_list, decap_map, best_gen_score, opt, replace_limit = None):
    # there are things that can be simplified. I dont know what, but there is
    end_pop = copy.deepcopy(decap_map)
    ref_score = best_gen_score
    base_score = opt.fail_score # use this variable for comparison purposes
    if not isinstance(replace_limit, int) and replace_limit is not None:
        raise ValueError('Replace limit must be a positive integer > 0 or Nonetype')
    elif isinstance(replace_limit, int) and replace_limit <= 0:
        raise ValueError('Replace limit must be a positive integer > 0 or Nonetype')


    # This section will extract solutions with unique zero positions. 1 solution for each unique zero solu set
    if ref_score >= base_score and np.count_nonzero(unique_best_list[0]) != opt.decap_ports:
        # If best solution has at least one empty port
        zeroes_pos = [[i for i, j in enumerate(unique_best_list[k]) if j == 0] for k in range(len(unique_best_list))]
        zeroes_copy = copy.deepcopy(zeroes_pos)
        associate_list = [i for i in range(len(zeroes_pos))]
        for zero_set in zeroes_copy:
            indices_repeat = [i for i, _ in enumerate(zeroes_pos) if _ == zero_set and zero_set]  # empty list is false
            if len(indices_repeat) > 1:
                remove_list = random.sample(indices_repeat, len(indices_repeat) - 1)
                for j in remove_list:  # if nothing in remove list, it shouldn't loop. Seemed that way from testing
                    zeroes_pos[j] = None
                    associate_list.remove(j)
        unique_zeroes_list = [unique_best_list[i] for i in associate_list]
        print(len(unique_zeroes_list), 'Unique Zero Positions')


        if isinstance(replace_limit, int) and replace_limit > len(unique_zeroes_list):
            # don't think I need to check if replace_limit is an int at this point.
            print('The replacement limit is greater than the number of unique zeroes solutions.'
                  ' Setting Replace limit to None')
            replace_limit = None
        print('Inserting best score solutions who have unique zero placements')
        print(unique_zeroes_list)

        if replace_limit is None:
            replace_pos = random.sample(list(range(len(unique_zeroes_list))), len(unique_zeroes_list))
            for index, sol in enumerate(random.sample(unique_zeroes_list, len(unique_zeroes_list))):
                end_pop[replace_pos[index]] = sol
            # for i in unique_zeroes_list:
            #         randPos = random.randrange(len(end_pop))
            #         end_pop[randPos] = i
        else:
            replace_pos = random.sample(list(range(len(unique_zeroes_list))), replace_limit)
            for index, sol in enumerate(random.sample(unique_zeroes_list, replace_limit)):
                end_pop[replace_pos[index]] = sol

    elif ref_score >= base_score and np.count_nonzero(unique_best_list[0]) == opt.decap_ports:
        # ie meeting impedance requirement, but has a decap in all ports
        print('There are no empty ports, however solution(s) meeting target impedance exists. '
              'To preserve diversity, only a max of 5 such solutions will be inserted.')
        for i in unique_best_list:
            if len(unique_best_list) >= 5:
                replace_sols_pos = random.sample([x for x,_ in enumerate(unique_best_list)],5)
                for j in replace_sols_pos:
                    randPos = random.randrange(len(end_pop))
                    end_pop[randPos] = unique_best_list[j]

            else:
                randPos = random.randrange(len(end_pop))
                end_pop[randPos] = i

    elif ref_score < opt.fail_score:  # ie impedance requirements not met
        print('No solution meeting impedance requirements. No elitism will occur.')
    return end_pop






def selBest(best_parent, rand_parent, decap_map):
    pass
    # Roulette might take too long
    if min(solu_scores) != opt.fail_score:  # not every score is bad
        best_scores = [i for i, x in enumerate(solu_scores) if x == min(solu_scores)] # get indices with best score
        best_parent = random.choice(best_scores) # should return the index of a parent with best score
        rand_parent = random.randrange(0,len(solu_scores)) # get index of a random parent
        while rand_parent == best_parent:
             rand_parent = random.randrange(0, len(solu_scores))
    else:
        best_parent = random.randrange(0, len(solu_scores))
        rand_parent = random.randrange(0, len(solu_scores))
        while rand_parent == best_parent:
            rand_parent = random.randrange(0, len(solu_scores))



#if best_gen_score < best_score:
            # I'm not too sure if this ever does anything. If best_gen_score is better, best_score takes on that value.
            #end_pop[random.randrange(len(mutated_pop))] = best_net
            # yea it doesn't do anything. When best_gen_score is better than the overall best_score, overall best_score
            # takes on the value of best_gen_score. That score is then forced to the next generation.
            # best_gen_score never less than best score


    # If the best score of a generation is equal to the best score overall, I will insert both the map
    # of the best overall score, and a random map of the best score of the generation, to the next generation
    #

    # if best_net not in end_pop:
    #     print('Hello There')
    #     randPos = random.randrange(len(mutated_pop))
    #     end_pop[randPos] = best_net
    #     if best_random_net not in end_pop:
    #         randPos2 = random.randrange(len(mutated_pop))
    #         while (randPos2 == randPos):
    #             randPos2 = random.randrange(len(mutated_pop))
    #         end_pop[randPos2] = best_random_net






'''
def calc_impedance(decap_map, decode_map, opt):
    # decap_map = the list of encoded numbers
    solu_net = pdn.connect_n_decap(decap_map, decode_map, opt) # generate completed networks provided solutions list
    solu_scores = []

    for i in range(len(solu_net)):
        zpdn = solu_net[i].z[:, 0, 0]  # returns impedance array
        if np.max((np.absolute(zpdn)))> opt.ztarget:
            solu_scores.append(100)   # if impedance target not met, give bad score
        else:
            solu_scores.append(len(solu_net[0][:]) - solu_net[i][:].count('None'))
            # Should append number of decaps
    return solu_net,solu_scores
'''

#if best_gen_score < best_score:
            # I'm not too sure if this ever does anything. If best_gen_score is better, best_score takes on that value.
            #end_pop[random.randrange(len(mutated_pop))] = best_net
            # yea it doesn't do anything. When best_gen_score is better than the overall best_score, overall best_score
            # takes on the value of best_gen_score. That score is then forced to the next generation.
            # best_gen_score never less than best score