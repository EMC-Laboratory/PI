import numpy as np
import random
import copy
import math
import ShapePDN as pdn1
import collections



def find_cutoff_freq(input_net_z, opt):
    cutoff_index = (np.greater(np.absolute(input_net_z), opt.ztarget)).tolist().index(1)
    return cutoff_index


def generate_chromosome(opt):
    return np.random.randint(opt.num_decaps + 1, size=(1, opt.decap_ports))


def add_zeroes(chromosome, num_zeroes=1):

    new_chromosome = copy.deepcopy(chromosome)
    if not isinstance(num_zeroes, int):
        raise TypeError('Number of Zeroes Added Must Be An Integer')
    replace_pts = random.sample(range(len(new_chromosome)), num_zeroes)
    for i in replace_pts:
        new_chromosome[i] = 0
    return new_chromosome

def add_zeroes2(chromosome, bulk_cap, num_zeroes=1):
    # adds up to a certain # of zeroes while accounting for zeroes already in chromosome
    new_chromosome = copy.deepcopy(chromosome)
    if not isinstance(num_zeroes, int):
        raise TypeError('Number of Zeroes Added Must Be An Integer')
    zero_pts = np.nonzero(chromosome == 0)[0]
    non_zero_pts = np.nonzero(chromosome)[0].tolist()
    if len(zero_pts) == num_zeroes:
        # print('No zeroes need to be added')
        pass
    elif len(zero_pts) < num_zeroes:
        replace_pts = random.sample(np.nonzero(chromosome)[0].tolist(), num_zeroes - len(zero_pts))
        #print(replace_pts, 'replace points')
        for i in replace_pts:
            new_chromosome[i] = 0
    else:
        replace_pts = random.sample(zero_pts.tolist(), len(zero_pts) - num_zeroes)
        #print(replace_pts, 'replace points')
        for i in replace_pts:
            #new_chromosome[i] = random.randrange(range(len(non_zero_pts)), bulk_cap + 1)
            to_replace = random.sample(range(len(non_zero_pts)),1)[0]
            new_chromosome[i] = chromosome[non_zero_pts[to_replace]]
    return new_chromosome



def add_number(chromosome, number_added, opt, total_added=1):
    # number added is the Number you want to add, eg 1, 2, 3
    # total added is the number of number_added you want to appear

    if not isinstance(total_added, int) or not isinstance(number_added, int):
        raise TypeError('Number to Add and Amount Added Must be An Integer')
    elif total_added > opt.decap_ports or number_added > opt.decap_ports:
        raise ValueError('Amount Added or Number to Add Must Be Less or Equal to total # of ports')

    new_chromosome = copy.deepcopy(chromosome)
    num_needed = total_added - np.count_nonzero(new_chromosome == number_added)
    replace_pts = []
    if num_needed > 0:
        replace_list = [i for i, j in enumerate(chromosome) if j != number_added]
        replace_pts = random.sample(replace_list, num_needed)
    for i in replace_pts:
        new_chromosome[i] = number_added
    return new_chromosome


def check_for_sol(decap_maps_z, opt):
    # Initial check of scores. See if any solutions happen to satisfy impedance targets from initial bunch.
    success_bool = False
    initial_sol_index = []

    for map_num, map_z in enumerate(decap_maps_z):  # each solution's z-parameters
        if np.count_nonzero(np.greater(np.absolute(map_z), opt.ztarget)) == 0:
            success_bool = True
            initial_sol_index = map_num
            break
    return success_bool, initial_sol_index  # Check if a solution satisfies target impedance.


def initial_sol_score(decap_maps_z, opt):
    decap_map_scores = [None] * len(decap_maps_z)

    for map_num, map_z in enumerate(decap_maps_z):

        min_z_ind = np.argmin(np.absolute(map_z))
        min_z = np.absolute(map_z[min_z_ind])
        max_z_ind = np.argmax(np.absolute(map_z))
        max_z = np.absolute(map_z[max_z_ind])

        # old scoring method
        # score_modifier = round((opt.ztarget[min_z_ind] / (opt.ztarget[min_z_ind] - min_z)) / max_z)
        # score_modifier = round((opt.ztarget[min_z_ind]/ (abs(opt.ztarget[min_z_ind] - min_z))) / (max_z - opt.ztarget[max_z_ind]))  <--- og scoring
        # print('Pass Mod =', score_modifier)
        # decap_map_scores[map_num] = base_score + score_modifier
        # decap_map_scores[map_num] = score_modifier

        # # Very good for 27 port, bad for 15 port
        # Score based on average impedance above target
        points_above = np.greater(np.absolute(map_z), opt.ztarget)
        points_above_index = [index for index, _ in enumerate(points_above) if _ == True]

        holder = 0  # holds running sum of the percentage above target over each point above
        z_holder = 0
        max_z_above = 0
        for index in points_above_index:
            percentage_greater = np.absolute(map_z[index]) / opt.ztarget[index]

            if np.absolute(map_z[index]) > max_z_above:
                max_z_above = np.absolute(map_z[index])
            z_holder = z_holder + np.absolute(map_z[index])

            holder = holder + percentage_greater

        average_percent_greater = holder / len(points_above_index)
        # z_holder_average = z_holder/len(points_above_index)
        # target_area = np.trapz(y=opt.ztarget, x=opt.freq)
        # solution_area = np.trapz(y=np.absolute(decap_maps_z[map_num]), x=opt.freq)

        solution_average = sum(np.absolute(map_z)) / len(map_z)
        holder2 = 0
        for i in np.absolute(map_z):
            if i > solution_average:
                holder2 = holder2 + i / solution_average
            else:
                holder2 = holder2 + solution_average / i
        average_from_center = holder2 / len(map_z)
        # print(average_from_center)
        decap_map_scores[map_num] = round(
            1 / (average_percent_greater - 1) + 1 / (average_from_center - 1) + 1 / (max_z_above - min_z))

    return decap_map_scores


def initial_sol_score2(decap_maps, decap_maps_z, opt, min_z_pt, shift_pt, bulk_full_z, min_full_z, cap_objs_z):
    decap_map_scores = [None] * len(decap_maps_z)


    min_full_mins = pdn1.find_mins(abs(min_full_z), local_only=False)
    furthest_min_f = opt.freq[min_full_mins[-1]]  # this freq is the most left the most pronounced local min can move

    bulk_min_z = min(abs(bulk_full_z))

    for map_num, map_z in enumerate(decap_maps_z):
        abs_map_z = np.abs(map_z)


        points_above = np.greater(abs_map_z, opt.ztarget)  # returns a true false array
        points_above_index = [index for index, _ in enumerate(points_above) if _ == True]

        holder = 0  # holds running sum of the percentage above target over each point above
        z_holder = 0
        max_z_above = 0

        for index in points_above_index:
            percentage_greater = np.absolute(map_z[index]) / opt.ztarget[index]

            if np.absolute(map_z[index]) > max_z_above:
                max_z_above = np.absolute(map_z[index])
            z_holder = z_holder + np.absolute(map_z[index])

            holder = holder + percentage_greater

        average_percent_greater = holder / len(points_above_index)

        ###### Low Frequency Modifiers #######
        sol_min_z = opt.freq[np.argmin(abs_map_z)]

        # Fit minimum point location
        # maybe range could be adaptive based on decap list
        correct_min_location = True

        if sol_min_z >= min_z_pt / 1.5 and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod = 2
        elif sol_min_z >= min_z_pt / 1.75 and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod = 1.5
        elif sol_min_z >= min_z_pt / 2 and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod = 1.25
        else:
            min_z_location_mod = 0
            correct_min_location = False

        # Fit minimum z point value (bring min_z up)
        # Questions:
        # How do I scale these limits
        # Should I make them around min z point or ideal min z point
        # Min z value should incrementally increase?
        if correct_min_location and np.count_nonzero(decap_maps[map_num]):
            min_z_value_mod = (min(abs_map_z) / bulk_min_z) / 2
            # if min(abs_map_z) >= 3 * bulk_min_z:
            #     min_z_value_mod = 2
            # elif min(abs_map_z) >= 2 * bulk_min_z:
            #     min_z_value_mod= 1.5
            # elif min(abs_map_z) >= 1.5 * bulk_min_z:
            #     min_z_value_mod= 1
            # else:
            #     min_z_value_mod = 0
        else:
            min_z_value_mod = 0

        #### Right Frequency Range Modifier #####

        # Slope into last point should be close as possible
        # Set last local min desired position
        mins = pdn1.find_mins(abs_map_z, local_only=False)

        if opt.freq[mins[-1]] > furthest_min_f:
            # the problem here is that the difference is very very marginal
            last_min_mod = opt.freq[mins[-1]] / furthest_min_f
        else:
            last_min_mod = 0


        # Value of Last Impedance Point
        if last_min_mod != 0 and abs_map_z[-1] <= opt.ztarget[-1]:
            if abs_map_z[-1] >= opt.ztarget[-1]:
                last_z_value_mod = 2
            elif abs_map_z[-1] >= opt.ztarget[-1] * .98:
                last_z_value_mod = 1.5
            elif abs_map_z[-1] >= opt.ztarget[-1] * .97:
                last_z_value_mod = 1.25
            else:
                last_z_value_mod = 1
        else:
            last_z_value_mod = 0
        # value change is super marginal, abs_map_z[-1] /opt.ztarget[-1] etc
        # last_z_value_mod = 1 + abs_map_z[-1]/opt.ztarget[-1] if last_min_mod != 0 else 0
        last_z_value_mod = abs_map_z[-1] / abs(min_full_z[-1]) if (last_min_mod != 0 and abs_map_z[-1] <= opt.ztarget[-1]) else 0
        #print('Last_z_value_mod is:', last_z_value_mod, 'With impedance:', abs_map_z[-1])



        # Fix center, reduce number of unique capacitors in center range
        # holder = decap_maps[map_num].copy()
        # holder = [i for i in holder if i > 0]
        # holder2 = [i for i in holder.copy() if i != min_cap and i != bulk_cap]
        #holder2 = [i for i in holder.copy() if i != min(holder) and i != max(holder)]
        #num_unique_center = len(set(holder2))
        # print('Caps in center =', holder2)
        # print('Unique caps in center=', num_unique_center)

        # Fix center, reduce excessive types of a capacitor within the center range
        # I stole part of it from https://www.geeksforgeeks.org/python-element-with-largest-frequency-in-list/
        # if len(set(holder2)) != 0:
        #     most_common_cap = max(set(holder2), key=holder2.count)
        #     num_common = holder2.count(most_common_cap)
        # else:
        #     most_common_cap = max(set(holder), key=holder.count)
        #     num_common = holder.count(most_common_cap)

        # print('Most common cap:', most_common_cap, 'with:', num_common)
        # print('Caps in center =', holder2)
        # print('Unique Caps in Center = ', num_unique_center)



        # score_multiplier = (multiplier * (last_z_multiplier + max_multiplier) - .5 * num_common - .5*num_unique_center if
        # multiplier * (last_z_multiplier + max_multiplier) - .5 * num_common - .5 * num_unique_center > 0 else 1)

        #score_multiplier = min_z_location_mod + min_z_value_mod + last_min_mod #+ last_z_value_mod
        score_multiplier =  (min_z_location_mod + min_z_value_mod + last_min_mod + last_z_value_mod) if min_z_location_mod != 0 else 0


        # print('Final shape modifier =', score_multiplier)

        decap_map_scores[map_num] = round( 1 / (average_percent_greater - 1) * (1+score_multiplier))
        #decap_map_scores[map_num] = ((1 + score_multiplier))

    return decap_map_scores



def EmptyPortSearch(decap_maps, decap_maps_z, opt, min_z_pt, shift_pt, bulk_full_z, min_full_z, cap_objs_z):
    decap_map_scores = [None] * len(decap_maps_z)


    min_full_mins = pdn1.find_mins(abs(min_full_z), local_only=False)
    furthest_min_f = opt.freq[min_full_mins[-1]]  # this freq is the most left the most pronounced local min can move

    bulk_min_z = min(abs(bulk_full_z))

    for map_num, map_z in enumerate(decap_maps_z):
        abs_map_z = np.abs(map_z)


        points_above = np.greater(abs_map_z, opt.ztarget)  # returns a true false array
        points_above_index = [index for index, _ in enumerate(points_above) if _ == True]

        holder = 0  # holds running sum of the percentage above target over each point above
        z_holder = 0
        max_z_above = 0

        for index in points_above_index:
            percentage_greater = np.absolute(map_z[index]) / opt.ztarget[index]

            if np.absolute(map_z[index]) > max_z_above:
                max_z_above = np.absolute(map_z[index])
            z_holder = z_holder + np.absolute(map_z[index])

            holder = holder + percentage_greater

        average_percent_greater = holder / len(points_above_index)

        ###### Low Frequency Modifiers #######
        sol_min_z = opt.freq[np.argmin(abs_map_z)]

        # Fit minimum point location
        # maybe range could be adaptive based on decap list
        correct_min_location = True

        if sol_min_z >= min_z_pt / 1.5 and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod = 2
        elif sol_min_z >= min_z_pt / 1.75 and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod = 1.5
        elif sol_min_z >= min_z_pt / 2 and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod = 1.25
        else:
            min_z_location_mod = 0
            correct_min_location = False

        # Fit minimum z point value (bring min_z up)
        # Questions:
        # How do I scale these limits
        # Should I make them around min z point or ideal min z point
        # Min z value should incrementally increase?
        if correct_min_location and np.count_nonzero(decap_maps[map_num]):
            min_z_value_mod = (min(abs_map_z) / bulk_min_z) / 2
            # if min(abs_map_z) >= 3 * bulk_min_z:
            #     min_z_value_mod = 2
            # elif min(abs_map_z) >= 2 * bulk_min_z:
            #     min_z_value_mod= 1.5
            # elif min(abs_map_z) >= 1.5 * bulk_min_z:
            #     min_z_value_mod= 1
            # else:
            #     min_z_value_mod = 0
        else:
            min_z_value_mod = 0

        #### Right Frequency Range Modifier #####

        # Slope into last point should be close as possible
        # Set last local min desired position
        mins = pdn1.find_mins(abs_map_z, local_only=False)

        if opt.freq[mins[-1]] > furthest_min_f:
            # the problem here is that the difference is very very marginal
            last_min_mod = opt.freq[mins[-1]] / furthest_min_f
        else:
            last_min_mod = 0


        # Value of Last Impedance Point
        if last_min_mod != 0 and abs_map_z[-1] <= opt.ztarget[-1]:
            if abs_map_z[-1] >= opt.ztarget[-1]:
                last_z_value_mod = 2
            elif abs_map_z[-1] >= opt.ztarget[-1] * .98:
                last_z_value_mod = 1.5
            elif abs_map_z[-1] >= opt.ztarget[-1] * .97:
                last_z_value_mod = 1.25
            else:
                last_z_value_mod = 1
        else:
            last_z_value_mod = 0
        # value change is super marginal, abs_map_z[-1] /opt.ztarget[-1] etc
        # last_z_value_mod = 1 + abs_map_z[-1]/opt.ztarget[-1] if last_min_mod != 0 else 0
        last_z_value_mod = abs_map_z[-1] / abs(min_full_z[-1]) if (last_min_mod != 0 and abs_map_z[-1] <= opt.ztarget[-1]) else 0
        #print('Last_z_value_mod is:', last_z_value_mod, 'With impedance:', abs_map_z[-1])



        # Fix center, reduce number of unique capacitors in center range
        # holder = decap_maps[map_num].copy()
        # holder = [i for i in holder if i > 0]
        # holder2 = [i for i in holder.copy() if i != min_cap and i != bulk_cap]
        #holder2 = [i for i in holder.copy() if i != min(holder) and i != max(holder)]
        #num_unique_center = len(set(holder2))
        # print('Caps in center =', holder2)
        # print('Unique caps in center=', num_unique_center)

        # Fix center, reduce excessive types of a capacitor within the center range
        # I stole part of it from https://www.geeksforgeeks.org/python-element-with-largest-frequency-in-list/
        # if len(set(holder2)) != 0:
        #     most_common_cap = max(set(holder2), key=holder2.count)
        #     num_common = holder2.count(most_common_cap)
        # else:
        #     most_common_cap = max(set(holder), key=holder.count)
        #     num_common = holder.count(most_common_cap)

        # print('Most common cap:', most_common_cap, 'with:', num_common)
        # print('Caps in center =', holder2)
        # print('Unique Caps in Center = ', num_unique_center)



        # score_multiplier = (multiplier * (last_z_multiplier + max_multiplier) - .5 * num_common - .5*num_unique_center if
        # multiplier * (last_z_multiplier + max_multiplier) - .5 * num_common - .5 * num_unique_center > 0 else 1)

        #score_multiplier = min_z_location_mod + min_z_value_mod + last_min_mod #+ last_z_value_mod
        score_multiplier =  (min_z_location_mod + min_z_value_mod + last_min_mod + last_z_value_mod) if min_z_location_mod != 0 else 0


        # print('Final shape modifier =', score_multiplier)

        decap_map_scores[map_num] = round( 1/(5*(average_percent_greater - 1)) * (1+score_multiplier))
        #decap_map_scores[map_num] = ((1 + score_multiplier))

    return decap_map_scores




def calc_score(decap_maps, decap_maps_z, opt, min_zero_map, min_z_pt,  bulk_full_z, min_full_z, distances, shift_f):


    decap_map_scores = [None] * len(decap_maps_z)
    min_decap_num = len(min_zero_map) - np.count_nonzero(min_zero_map)
    new_min_zero_map = copy.deepcopy(min_zero_map)


    min_full_mins = pdn1.find_mins(abs(min_full_z), local_only=False)
    furthest_min_f = opt.freq[min_full_mins[-1]]  # this freq is the most left the most pronounced local min can move


    bulk_min_z = min(abs(bulk_full_z))


    #bulk_f_ind = np.argmin(min(abs(bulk_full_z)))

    #min_z_pt = opt.freq[bulk_f_ind]

    for map_num, map_z in enumerate(decap_maps_z):  # iterates through each solution in population
        num_empty = len(decap_maps[map_num]) - np.count_nonzero(decap_maps[map_num])
        abs_map_z = np.absolute(map_z)

        if np.count_nonzero(np.greater(abs_map_z, opt.ztarget)) == 0:
            # if solution does meet target, check if it has the smallest # of caps
            if num_empty > min_decap_num:
                new_min_zero_map = decap_maps[map_num]

            # Num of Decaps score
            decap_score = num_empty + 1

            ###### Minimum point modifier  #######
            sol_min_z = opt.freq[np.argmin(abs_map_z)]

            # Fit minimum point location
            correct_min_location = True

            if sol_min_z >= min_z_pt / 1.5 and sol_min_z <= min_z_pt * 1.5:
                min_z_location_mod = 2
            elif sol_min_z >= min_z_pt / 1.75  and sol_min_z <= min_z_pt * 1.5:
                min_z_location_mod  = 1.5
            elif sol_min_z >= min_z_pt / 2  and sol_min_z <= min_z_pt * 1.5:
                min_z_location_mod  = 1.25
            else:
                min_z_location_mod  = 0
                correct_min_location = False
            #print('here', min_z_location_mod)

            # print('Min z:', min(abs_map_z), 'at f', opt.freq[np.argmin(abs_map_z)])
            # print('min z location mod', min_z_location_mod)

            # Fit minimum z point value (bring min_z up)
            # Questions:
            # How do I scale these limits
            # Should I make them around min z point or ideal min z point
            # Min z value should incrementally increase? The bounds rn aren't good enough, crosses to different
            # cap ranges (but i think this is unavoidable)
            # bring the min z up should have the effect of reducing the number of bulk cap
            # may tell you something about port location too, idk

            if correct_min_location and np.count_nonzero(decap_maps[map_num]):
                min_z_value_mod = (min(abs_map_z)/bulk_min_z)/2
                # if min(abs_map_z) >= 3 * bulk_min_z:
                #     min_z_value_mod = 2
                # elif min(abs_map_z) >= 2 * bulk_min_z:
                #     min_z_value_mod= 1.5
                # elif min(abs_map_z) >= 1.5 * bulk_min_z:
                #     min_z_value_mod= 1
                # else:
                #     min_z_value_mod = 0
            else:
                min_z_value_mod = 0
            #min_z_value_mod = 2 * (1 + min(abs_map_z)/opt.ztarget[0])
            # print('Min z:', min(abs_map_z), 'at f', opt.freq[np.argmin(abs_map_z)])
            # print('Value mulitplier', min_z_value_mod)

            #### Right Frequency Range Modifier #####

            # Slope into last point should be close as possible
            # Set last local min desired position
            mins = pdn1.find_mins(abs_map_z, local_only=False)

            if opt.freq[mins[-1]] > furthest_min_f and opt.freq[mins[-1]] <= shift_f:
                # the problem here is that the difference is very very marginal
                last_min_mod = opt.freq[mins[-1]]/furthest_min_f
            else:
                last_min_mod = 0

            # Value of Last Impedance Point
            # needs work, the effect is marginal and it depends on the z target
            if last_min_mod != 0:
                if abs_map_z[-1] >= opt.ztarget[-1]*.99:
                    last_z_value_mod= 2
                elif abs_map_z[-1] >= opt.ztarget[-1]* .98:
                    last_z_value_mod= 1.5
                elif abs_map_z[-1] >= opt.ztarget[-1]* .97:
                    last_z_value_mod= 1.25
                else:
                    last_z_value_mod = 1
            else:
                last_z_value_mod = 0
            # value change is super marginal, abs_map_z[-1] /opt.ztarget[-1] etc
            #last_z_value_mod = 1 + abs_map_z[-1]/opt.ztarget[-1] if last_min_mod != 0 else 0
            last_z_value_mod = abs_map_z[-1] / abs(min_full_z[-1])  if last_min_mod != 0 else 0


            #### Testing Idea of Considering Port Location, really rough
            #print(distances)
            bulk_distance = distances[0:math.floor(len(distances)/2)]
            min_distance = distances[math.floor(len(distances)/2)+1:len(distances)]
            decap_count = 0
            bulk_cap_placement = False

            for i in bulk_distance:
                if decap_maps[map_num][i-1] == 8:
                   bulk_cap_placement = True

            for i in min_distance:
                if decap_maps[map_num][i-1] < 8 and decap_maps[map_num][i-1] > 0:
                    decap_count = decap_count + 1

            #print('test', decap_count)
            test_multiplier = decap_count/2 if bulk_cap_placement else 0

            # Assign Score
            # [9, 8, 3, 10, 7, 6, 11, 1, 5, 2, 14, 4, 12, 13]

            # score = (decap_score * ( min_z_location_mod + min_z_value_mod + last_min_mod + last_z_value_mod + test_multiplier)
            #        if decap_score * ( min_z_location_mod + min_z_value_mod + last_min_mod + last_z_value_mod +test_multiplier) > 2 else 2)


            score = decap_score * ( min_z_location_mod + min_z_value_mod + last_min_mod +last_z_value_mod) \
                   if decap_score * ( min_z_location_mod + min_z_value_mod + last_min_mod +last_z_value_mod)  > 2 else 2



            # score = (decap_score * (min_z_location_mod + min_z_value_mod + last_min_mod) #+ last_z_value_mod
            #          if decap_score * (
            #             min_z_location_mod + min_z_value_mod + last_min_mod ) > 2 else 2)


            #print('Here', score)
            decap_map_scores[map_num] = score

        else:
            score = 1
            replace_chance = random.randint(0, 1)

            if not replace_chance:
                decap_maps[map_num] = generate_chromosome(opt)
                bulk_cap = 2
                decap_maps[map_num] = add_zeroes2(decap_maps[map_num], bulk_cap, min_decap_num)
                decap_map_scores[map_num] = score
            decap_map_scores[map_num] = score

    return decap_map_scores, new_min_zero_map

def basic_score(decap_maps, decap_maps_z, opt):
    decap_map_scores = [None] * len(decap_maps_z)

    for map_num, map_z in enumerate(decap_maps_z):  # iterates through each solution in population
        num_empty = len(decap_maps[map_num]) - np.count_nonzero(decap_maps[map_num])
        abs_map_z = np.abs(map_z)
        if np.count_nonzero(np.greater(abs_map_z, opt.ztarget)) == 0:

            decap_score = 5 + num_empty

        else:

            decap_score = np.max((abs_map_z - opt.ztarget) / opt.ztarget)
            decap_score = 1 - np.max((abs_map_z - opt.ztarget) / opt.ztarget) if np.max((abs_map_z - opt.ztarget) / opt.ztarget) < 1 \
                            else np.max((abs_map_z - opt.ztarget)/opt.ztarget) - 1

        decap_map_scores[map_num] = decap_score

    return decap_map_scores

def calc_score2(decap_maps, decap_maps_z, opt, min_zero_map, min_z_pt,  bulk_full_z, min_full_z, distances):

    # i said the capacitor location doesn't really matter to the shape
    # but i kept anyway because me == Idiot. 9/19

    decap_map_scores = [None] * len(decap_maps_z)
    new_min_zero_map = copy.deepcopy(min_zero_map)


    min_full_mins = pdn1.find_mins(abs(min_full_z), local_only=False)
    furthest_min_f = opt.freq[min_full_mins[-1]]  # this freq is the most left the most pronounced local min can move
    bulk_min_z = min(abs(bulk_full_z))


    bulk_f_ind = np.argmin(min(abs(bulk_full_z)))


    for map_num, map_z in enumerate(decap_maps_z):  # iterates through each solution in population
        abs_map_z = np.absolute(map_z)

        ###### Minimum point modifier  #######
        sol_min_z = opt.freq[np.argmin(abs_map_z)]

        # Fit minimum point location
        # maybe range could be adaptive based on decap list
        correct_min_location = True

        min_z_pt = opt.freq(bulk_f_ind)


        if sol_min_z >= min_z_pt / 1.5 and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod = 2
        elif sol_min_z >= min_z_pt / 1.75  and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod  = 1.5
        elif sol_min_z >= min_z_pt / 2  and sol_min_z <= min_z_pt * 1.5:
            min_z_location_mod  = 1.25
        else:
            min_z_location_mod  = 0
            correct_min_location = False


        if correct_min_location:
            min_z_value_mod = (min(abs_map_z)/bulk_min_z)/2
            # if min(abs_map_z) >= 3 * bulk_min_z:
            #     min_z_value_mod = 2
            # elif min(abs_map_z) >= 2 * bulk_min_z:
            #     min_z_value_mod= 1.5
            # elif min(abs_map_z) >= 1.5 * bulk_min_z:
            #     min_z_value_mod= 1
            # else:
            #     min_z_value_mod = 0
        else:
            min_z_value_mod = 0

        #### Right Frequency Range Modifier #####

        # Slope into last point should be close as possible
        # Set last local min desired position
        mins = pdn1.find_mins(abs_map_z, local_only=False)

        if opt.freq[mins[-1]] > furthest_min_f:
            # the problem here is that the difference is very very marginal
            last_min_mod = opt.freq[mins[-1]]/furthest_min_f
        else:
            last_min_mod = 0

        # Value of Last Impedance Point
        # needs work, the effect is marginal and it depends on the z target
        if last_min_mod != 0:
            if abs_map_z[-1] >= opt.ztarget[-1]*.99:
                last_z_value_mod= 2
            elif abs_map_z[-1] >= opt.ztarget[-1]* .98:
                last_z_value_mod= 1.5
            elif abs_map_z[-1] >= opt.ztarget[-1]* .97:
                last_z_value_mod= 1.25
            else:
                last_z_value_mod = 1
        else:
            last_z_value_mod = 0
        # value change is super marginal, abs_map_z[-1] /opt.ztarget[-1] etc
        #last_z_value_mod = 1 + abs_map_z[-1]/opt.ztarget[-1] if last_min_mod != 0 else 0
        last_z_value_mod = abs_map_z[-1] / abs(min_full_z[-1])  if last_min_mod != 0 else 0

        #### number of points below
        pts_below = np.count_nonzero(np.greater(opt.ztarget[0:195],abs_map_z[0:195]))/10
        pts_below_high_f = np.count_nonzero(np.greater(opt.ztarget[195:opt.nf],abs_map_z[195:opt.nf]))

        pts_score = pts_below + pts_below_high_f



        # score = pts_score * ( min_z_location_mod + min_z_value_mod + last_min_mod +last_z_value_mod) if \
        # pts_below * (min_z_location_mod + min_z_value_mod + last_min_mod + last_z_value_mod) > 0 else 1

        score = pts_score  if pts_below  > 0 else 1

        decap_map_scores[map_num] = score

    return decap_map_scores, new_min_zero_map


def seed_score(decap_maps, decap_maps_z,opt, shift_f):

    # Scoring to try and find capacitors that can yield a global minimum solution
    decap_map_scores = [None] * len(decap_maps_z)
    shift_index = np.nonzero(opt.freq == shift_f)[0]


    # For now lets assume constant region, or peaks will be in constant region based on decaps chosen
    # lets do the slope change later

    for map_num,i in enumerate(decap_maps_z):

        abs_map_z = np.abs(i)

        #Gets indices of peaks that occur before the shift index
        maxs = pdn1.find_maxs(abs_map_z, local_only= True, last_z= False)
        maxs_z = [abs_map_z[i] for i in maxs if i < shift_index]
        maxs_z_a = np.array(maxs_z)
        reference = np.array([opt.ztarget[i] for i in maxs if i < shift_index])

        # Gets the indexes of peaks above the target, before the shift index
        peaks_above = np.nonzero(np.greater(maxs_z_a, reference))[0]
        peaks_above = peaks_above.astype(int)

        # print('maxs', maxs)
        # print('maxs z', maxs_z_a)
        # print('refernce', reference)
        # print('peaks_above', peaks_above)
        # print('abs map z', abs_map_z)
        # print('decap map', decap_maps[map_num])

        # Remove indices of all peaks above target but before shift index
        peaks_above = np.flip(peaks_above) # remove from larger indicies first
        if len(peaks_above) != 0:
            for i in peaks_above:
                maxs_z_a = np.delete(maxs_z_a,i)
                reference = np.delete(reference,i)


        # Calculate the average of the peaks that are below the target

        target_met = np.count_nonzero(np.greater(abs_map_z, opt.ztarget))

        if len(maxs_z_a) != 0:
            average_z_peaks = np.sum(maxs_z_a)/np.shape(maxs_z_a)[0]
            # print('z')
            # print(maxs_z_a)
            # print(np.sum(maxs_z_a))
            # print(np.shape(maxs_z_a)[0])
            # print(np.sum(maxs_z_a)/np.shape(maxs_z_a)[0])
            reference_average = np.sum(reference)/np.shape(reference)[0]
            # print('ref')
            # print(reference)
            # print(np.sum(reference))
            # print(np.shape(reference)[0])
            # print(np.sum(reference)/np.shape(reference)[0])
            # print('subtraction')
            # print(reference_average - average_z_peaks)

            decap_map_scores[map_num] = 1 / (reference_average - average_z_peaks)

        else:
            decap_map_scores[map_num] = 1

        # else:
        #     # if a peak is above target
        #     average_z_peaks = sum(maxs_z) / len(maxs_z)
        #     reference_average = sum(reference) / len(reference)
        #     decap_map_scores[map_num] = .1 / (max(reference_average, average_z_peaks) - min(reference_average,average_z_peaks))
        #     print('Failed', average_z_peaks, reference_average)

    return decap_map_scores

def get_min_map(min_map, test_map):
    test_min_map = copy.deepcopy(min_map)
    if len(test_min_map) == 0 or (len(test_min_map) - np.count_nonzero(test_min_map)) < (
            len(test_map) - np.count_nonzero(test_map)):
        test_min_map = test_map

    return test_min_map


def roulette(decap_maps_scores, decap_maps):
    # Use roulette to generate new population
    # Roulette might take too long
    selected_parents = copy.deepcopy(decap_maps)
    scores_copy = copy.deepcopy(decap_maps_scores)




    total_score = sum(scores_copy)
    fitness = [i / total_score for i in scores_copy]
    roulette = np.cumsum(fitness)

    for i in range(len(decap_maps)):
        rand_event = random.random()
        for j in range(len(roulette)):
            if rand_event < roulette[j]:  # roulette wheel to calculate next generation
                selected_parents[i] = decap_maps[j]
                break
    return selected_parents


def cross_over(decap_maps, Population, opt):
    cross_pop = copy.deepcopy(decap_maps)  # Holder for cross over population

    cross_parents = [x for x, _ in enumerate(decap_maps) if random.random() < Population.crossoverRate]
    # get indices for parents that will crossover

    random.shuffle(cross_parents)
    # Mix order of indices up

    if len(cross_parents) >= 2:  # Two parents at least must exist for crossover to be done
        for i, parent_pos in enumerate(cross_parents):

            # can prob simplify, visually at least, with enumerate() later
            # Yep on review, this is written real stupidly

            point = random.randrange(1, opt.decap_ports - 1)  # Cross point designation
            # I did not want the end point to be the cross point designations because you just end up
            # copying an entire solution in.
            # however, that means it is hard for the endpoints to have diverse values


            left_parent = random.randrange(1, 3)  # 1 = i'th parent is on left. 2 = i+1 parent in on left
            if i + 1 == len(cross_parents):
                if left_parent == 1:
                    new_chromo = [decap_maps[parent_pos][j] for j in range(point)]
                    new_chromo = new_chromo + [decap_maps[cross_parents[0]][j] for j in range(point, opt.decap_ports)]
                    cross_pop[parent_pos] = new_chromo

                else:  # the i + 1 parent is on the left
                    new_chromo = [decap_maps[cross_parents[0]][j] for j in range(point, opt.decap_ports)]
                    new_chromo = new_chromo + [decap_maps[parent_pos][j] for j in range(point)]
                    cross_pop[parent_pos] = new_chromo
            else:
                if left_parent == 1:
                    new_chromo = [decap_maps[parent_pos][j] for j in range(point)]
                    new_chromo = new_chromo + [decap_maps[cross_parents[i + 1]][j] for j in
                                               range(point, opt.decap_ports)]
                    cross_pop[parent_pos] = new_chromo
                else:
                    new_chromo = [decap_maps[cross_parents[i + 1]][j] for j in range(point, opt.decap_ports)]
                    new_chromo = new_chromo + [decap_maps[parent_pos][j] for j in range(point)]
                    cross_pop[parent_pos] = new_chromo

    # This is the much, much simpler way
    # if map_num == len(parents):
    #     break
    # else:
    #     left_parent = random.randint(0, 1)  # if left parent = 0, the [map_num] gene used for cutoff
    #     # if left parent = 1, the [map_num+1] gene used for cutoff
    #     break_point = random.randint(1, opt.decap_ports - 1)
    #     # point to perform crossover. Don't use endpoints for it
    #
    #     if not left_parent:
    #         child = parents[map_num][:break_point] + parents[map_num + 1][break_point:]
    #     else:
    #         child = parents[map_num + 1][:break_point] + parents[map_num][break_point:]
    #     new_population.append(child)

    return cross_pop


def cross_over2(decap_maps, decap_map_scores, opt, Population, min_zero_map):
    # HAS AN ISSUE IF ALL PARENTS ARE GOOD PARENTS

    # instead of cross overing good parents with good parents (good with good scores)
    # try cross overing good parents with bad parents (good with bad score)
    # bad and good are comparitive

    # still think local mins are an issue here, less so I think for then with cross_over
    # the endpoints at least, don't change often/hard to change
    cross_pop = copy.deepcopy(decap_maps)  # Holder for cross over population

    # Hypothetically, solutions that don't meet target can end up in good parents if weight in fail solutions
    # not adjusted

    # Right now, its written slightly wrong where the 'bad_parents' can absorb solutions from the good
    # parents section. Due to how its written, need to fix
    # Also potential chance for bad parents to be empty

    good_scores_inds = [ind for ind, i in enumerate(decap_map_scores) if (i >= 1 * max(decap_map_scores))]
    # print('good scores inds', good_scores_inds)
    good_parents = []  # holder for unique maps of good parents
    good_parents_inds = []  # holder for indices of unique good parents

    for i in good_scores_inds:
        if not any(np.array_equal(decap_maps[i], x) for x in good_parents):  # probably more elegant way to do this
            good_parents.append(decap_maps[i])  # used to check for uniqueness
            good_parents_inds.append(i)

    good_parents_total_fit = sum([decap_map_scores[i] for i in good_parents_inds])
    good_parents_fit = [decap_map_scores[i] / good_parents_total_fit for i in good_parents_inds]

    # get solutions marked as 'bad' or not good enough score
    bad_scores_inds = [i for i in range(Population.population) if i not in good_parents_inds]
    bad_parents = []  # holder for unique maps of bad parents
    bad_parents_inds = []  # holder for indices of unique bad parents
    for i in bad_scores_inds:
        if not any(np.array_equal(decap_maps[i], x) for x in bad_parents):
            if not any(np.array_equal(decap_maps[i], y) for y in good_parents):
                bad_parents.append(decap_maps[i])
                # used to check for uniqueness
                # should also check that they aren't in good parents. probably
                bad_parents_inds.append(i)

    bad_parents_total_fit = sum([decap_map_scores[i] for i in bad_parents_inds])
    bad_parents_fit = [decap_map_scores[i] / bad_parents_total_fit for i in bad_parents_inds]

    # print('Bad parents:', [decap_maps[i] for i in bad_parents_inds])
    # print('Good parents:', [decap_maps[i] for i in good_parents_inds])

    num_replace = len(bad_scores_inds)
    # what i want to do is replace all the 'bad' score solutions but I only want to do crossover using unique
    # good and bad sols to increase variety
    children = []

    for i in range(num_replace):
        good_parent_ind = []
        bad_parent_ind = []
        # for good parent
        roulette = np.cumsum(good_parents_fit)
        rand_event = random.random()
        for j in range(len(roulette)):
            if rand_event < roulette[j]:  # roulette wheel to calculate next generation
                good_parent_ind = good_parents_inds[j]
                break

        # for bad parent
        roulette2 = np.cumsum(bad_parents_fit)
        rand_event = random.random()
        for j in range(len(roulette2)):
            if rand_event < roulette2[j]:  # roulette wheel to calculate next generation
                bad_parent_ind = bad_parents_inds[j]
                break

        point = random.randrange(1, opt.decap_ports - 1)
        # say there's 14 ports, [1:14] or [0:13], this picks crossover pt to be in [2:13] or [1:12]
        # to avoid having a child identical to parent
        port_range = [i for i in range(opt.decap_ports)]

        left_parent = random.randrange(2)
        # 0 = good parent on 'left', 1 = bad parent on left

        # does crossover
        if left_parent == 0:
            new_chromo = [decap_maps[good_parent_ind][j] for j in port_range[0:point]]
            new_chromo = new_chromo + [decap_maps[bad_parent_ind][j] for j in port_range[point:]]
        else:
            new_chromo = [decap_maps[bad_parent_ind][j] for j in port_range[0:point]]
            new_chromo = new_chromo + [decap_maps[good_parent_ind][j] for j in port_range[point:]]

        shuffle_chance = random.randrange(5)
        if shuffle_chance == 0:
            random.shuffle(new_chromo)
        children.append(new_chromo)

    print('Children:', children)
    for ind, chromosome in enumerate(children):
        # chromosome = add_zeroes(chromosome, opt.decap_ports - np.count_nonzero(min_zero_map))
        # add_zeroes breaks if its from before a solution is found
        cross_pop[bad_scores_inds[ind]] = chromosome  # replace bad members
    return cross_pop


def two_point_cross(decap_map, popu, opt):
    cross_pop = copy.deepcopy(decap_map)
    cross_parents = [x for x, _ in enumerate(decap_map) if random.random() < popu.crossoverRate]
    random.shuffle(cross_parents)
    if len(cross_parents) >= 2:
        for i, parent_pos in enumerate(cross_parents):
            left_parent = random.randrange(0, 2)  # pick left parent
            # Parents = [i, i+1], if left parent, take genes starting with i. If right parent, start with i + 1 genes
            P1 = random.randrange(1, opt.decap_ports - 1)  # cross point 1
            P2 = random.randrange(1, opt.decap_ports - 1)  # cross point 2
            while (P2 == P1):
                P2 = random.randrange(1, opt.decap_ports)
            leftP = min(P1, P2)
            rightP = max(P1, P2)

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
                    cross_pop[parent_pos] = (new_chromo +
                                             [decap_map[cross_parents[i + 1]][j] for j in
                                              range(rightP, opt.decap_ports)])
    return cross_pop


def uniform_crossover(decap_map, popu, opt):
    cross_pop = copy.deepcopy(decap_map)
    cross_parents = [x for x, _ in enumerate(decap_map) if random.random() < popu.crossoverRate]
    # from parents gotten from roulette, pick parents  Inbreeding?
    # random.shuffle(cross_parents)

    if len(cross_parents) >= 2:
        for index, parent_pos in enumerate(cross_parents):
            # index = indices of cross_parents. Parent_pos = parent positions in decap_map
            gene_distri = [random.randrange(2) for _ in range(opt.decap_ports)]
            # 0 will represent the i'th parent, 1 will represent the i + 1 parent. values in gene_distri will tell you
            # what genes to take from each parent

            if index + 1 == len(cross_parents):  # if on the last index of cross_parents
                for gene_pos, parent in enumerate(gene_distri):  # for each gene, and the parent you take it from
                    cross_pop[parent_pos][gene_pos] = (decap_map[parent_pos][gene_pos] if not parent else
                                                       decap_map[cross_parents[0]][gene_pos])
                    # cross_pop[chromo being overwriteen][gene being overwritten]
            else:
                for gene_pos, parent in enumerate(gene_distri):  # for each gene, and the parent you take it from
                    cross_pop[parent_pos][gene_pos] = (decap_map[parent_pos][gene_pos] if not parent else
                                                       decap_map[cross_parents[index + 1]][gene_pos])
    return cross_pop


def mutation(Population, cross_population, opt, initial_sol_found, added_zeros=1):
    # I can definitely simplify this entire function into a form like gene_shakeup. But not a huge priority
    if not initial_sol_found:
        added_zeros = 0

    num_mut = int(Population.mutationRate * opt.decap_ports * Population.population)
    # lazy way of rounding down to an int for the # of mutations
    mutated_pop = copy.deepcopy(cross_population)

    if num_mut == 0:
        return mutated_pop  # no mutations occur in this case
    for i in range(num_mut):
        ran_chromosome = random.randrange(len(cross_population))  # get random chromosome
        ran_gene = random.randrange(opt.decap_ports)  # get a random gene
        rand_decap = random.randrange(Population.numDecaps + added_zeros)
        if rand_decap > (Population.numDecaps - 1):
            rand_decap = 0
        while rand_decap == mutated_pop[ran_chromosome][ran_gene]:
            rand_decap = random.randrange(Population.numDecaps + added_zeros)
            if rand_decap > (Population.numDecaps - 1):
                rand_decap = 0
        mutated_pop[ran_chromosome][ran_gene] = rand_decap
        if not initial_sol_found and rand_decap == 0:
            mutated_pop[ran_chromosome][ran_gene] = random.randrange(1, opt.num_decaps)
    return mutated_pop

def mutation2(Population, cross_population, opt):
    # I can definitely simplify this entire function into a form like gene_shakeup. But not a huge priority

    mutated_population = copy.deepcopy(cross_population)
    mutation_rate = Population.mutationRate
    for index, map in enumerate(cross_population):
        for i, _ in enumerate(map):
            if random.random() <= mutation_rate:
                previous_value = map[i]
                replace_cap = random.randrange(0,opt.num_decaps+1)
                while previous_value == replace_cap:
                    replace_cap = random.randrange(0, opt.num_decaps + 1)
                mutated_population[index][i] = replace_cap

    return mutated_population

def seed_mutation(Population, cross_population, min_cap, bulk_cap):

    mutated_population = copy.deepcopy(cross_population)
    mutation_rate = Population.mutationRate
    for index, map in enumerate(cross_population):
        for i, _ in enumerate(map):
            if random.random() <= mutation_rate:
                previous_value = map[i]
                replace_cap = random.randrange(min_cap,bulk_cap+1)
                while previous_value == replace_cap:
                    replace_cap = random.randrange(min_cap, bulk_cap + 1)
                mutated_population[index][i] = replace_cap

    return mutated_population




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


def twin_removal(decap_maps, opt):
    new_maps = copy.deepcopy(decap_maps)
    for i, j in enumerate(decap_maps):
        duplicates = []
        for index, map in enumerate(decap_maps):
            if j.all() == map.all():
                duplicates.append(index)
        if len(duplicates) > 1:
            for d in duplicates:
                if d != i:
                    new_maps[d] = [random.randrange(opt.num_decaps + 1) for i in range(len(decap_maps[0]))]
    return new_maps


def chromosome_shakeup(decap_maps):
    new_gene_layout = copy.deepcopy(decap_maps)
    for i in new_gene_layout:
        random.shuffle(i)
    return new_gene_layout


def gene_shakeup(decap_map, best_indices, opt, best_score, zero_bias=0):
    # as it currently is, if the best score is the fail score, the genes won't be shaken up, they'll be kept
    # exactly the same.

    # this works, but is written in stupidly. Will change later

    shaken_map = copy.deepcopy(decap_map)
    if zero_bias < 0:
        print('zero_bias set as less than 0 which is invalid. Setting zero_bias to 0')
        zero_bias = 0
    print('Test:', zero_bias)
    for i in range(len(decap_map)):  # might be able to simplify this
        if best_score != opt.fail_score and i not in best_indices:  # all maps with scores worse than the best score
            shaken_map[i] = np.random.randint(opt.num_decaps + zero_bias, size=(1, opt.decap_ports))
            # shake up gene pool by creating new random chromos for stagnated populations
            # new random chromos will overwrite only those whose score is wirse then the best score
            for j in range(len(shaken_map[i])):
                # can't use for j in shaken_map[i] i dont think because i need the actual index.
                if shaken_map[i][j] > (opt.num_decaps - 1):
                    shaken_map[i][j] = 0
        elif best_score == opt.fail_score:  # if stagnate on the fail score, replace everything
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


def elitism(best_score, best_map, decap_maps, decap_map_scores, initial_sol_found, min_zero_map, other_scores=True):

    # Need to fix for finding initial solution

    final_pop = copy.deepcopy(decap_maps)

    print('Elitims', best_score)
    # Two elitism opitions, one for if an initial solution is found or not
    if other_scores is True and initial_sol_found is True:
        best_indices = [index for index, score in enumerate(decap_map_scores) if
                        score >= (best_score * 1) or score >= (max(decap_map_scores) * 1)]
    else:
        best_indices = [index for index, score in enumerate(decap_map_scores) if
                        (score == best_score or score == max(decap_map_scores))]

    # Determine the rough number of solutions that could be replaced via elitism
    #max_allowed_copies = 5 if initial_sol_found is True else math.floor(len(decap_maps)/4)  # changed this to /4 9/16/20
    max_allowed_copies = 5

    if len(best_indices) < max_allowed_copies:

        # Get members of population who will be overwritten
        replace_indices = random.sample(range(len(final_pop)), max_allowed_copies - len(best_indices))

        # shuffle the indicies.
        random.shuffle(replace_indices)

        # if there are more indices to be replaced than there are indicies for the best solutions,
        # take only enough indicies to replacea
        if len(replace_indices) > len(best_indices):
            replace_indices = replace_indices[0:len(best_indices)]

        # from the best performing solutions, select what solutions will be used to replace
        indexs_to_replace = random.sample(best_indices, len(replace_indices))

        # Beginning replacement
        for pos, index in enumerate(replace_indices):
            final_pop[index] = decap_maps[indexs_to_replace[pos]]
            # print(final_pop[index], 'Inserted')

        print('Elitism Occured')
        print('Scores Added in:', [decap_map_scores[i] for i in indexs_to_replace])

    else:
        print('Elitism Did Not Occur. Enough high scoring sols in Population.', len(best_indices), 'copies present.  '
                                                                                                'The limit is',
              max_allowed_copies)

    if max(decap_map_scores) < best_score:
        print('Inserting Overall Highest Scoring Solution to Ensure They Are Not Lost')
        for i in range(1):
            final_pop[random.randrange(len(final_pop))] = best_map  # make sure best score overall is not lost

    # Ensure min decap solution isn't lost

    if initial_sol_found:
        print('Inserting Minimal Decap Solution to Ensure They Are Not Lost')
        for i in range(1):
            pass
            #final_pop[random.randrange(len(final_pop))] = min_zero_map

    return final_pop


def elitism2(unique_best_list, decap_map, best_gen_score, opt, replace_limit=None):
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


def elitism3(unique_best_list, decap_map, best_gen_score, opt, replace_limit=None):
    # there are things that can be simplified. I dont know what, but there is
    end_pop = copy.deepcopy(decap_map)
    ref_score = best_gen_score
    base_score = opt.fail_score  # use this variable for comparison purposes
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
                replace_sols_pos = random.sample([x for x, _ in enumerate(unique_best_list)], 5)
                for j in replace_sols_pos:
                    randPos = random.randrange(len(end_pop))
                    end_pop[randPos] = unique_best_list[j]

            else:
                randPos = random.randrange(len(end_pop))
                end_pop[randPos] = i

    elif ref_score < opt.fail_score:  # ie impedance requirements not met
        print('No solution meeting impedance requirements. No elitism will occur.')
    return end_pop


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

# Old scoring
# base_score = (opt.decap_ports - np.count_nonzero(decap_maps[map_num]) + opt.no_empty_port_score if
#                             np.count_nonzero(decap_maps[map_num]) != opt.decap_ports else opt.no_empty_port_score)
# num_empty = opt.decap_ports - np.count_nonzero(decap_maps[map_num]) + 1  # num empty ports. plus 1 for if no empty ports
# score_modifier =   (1/min_z + min_z_ind)
# score_modifier = round((min_z / (opt.ztarget[min_z_ind] - min_z)) / difference)
# decap_map_scores[map_num] = base_score + score_modifier
# decap_map_scores[map_num] = round(score_modifier) * base_score)
# decap_map_scores[map_num] = score_modifier * num_empty
# decap_map_scores[map_num] = round(score_modifier  * num_empty)  # multiplied by num empty still has that potential barrier effect

# percentage close scoring
# num_empty = opt.decap_ports - np.count_nonzero(decap_maps[map_num])
# sum_percentage_below = 0
# for i in range(opt.nf):
#     sum_percentage_below = sum_percentage_below + np.absolute(map_z[i])/opt.ztarget[i]  # returns percentage below target impedance is for that point
# average_p_below = sum_percentage_below/opt.nf
# score = (num_empty + 1)/(1-average_p_below)
# decap_map_scores[map_num] = round(score)
# have to consider possibility that initially, 1/x score won't be very high initially
# AHHHHHHHHH
# [3 2 4 8 3 2 5 2 4 2 0 0 0 0 0 0 1 0 0 0 0 0 0 0 2 0] can give 12 decap solution



