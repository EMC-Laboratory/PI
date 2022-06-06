import numpy as np
from math import pi
import ShapePDN as pdn
import time
from scipy.signal import argrelextrema



######### Functions for Generating Groups for a Particular Target Impedance #########
def identify_R_or_RL_Target(z_target):
    ## This function will tell you if the impepdance target is R or RL type
    ## Return the type as a string
    z_target_copy = np.copy(z_target)
    R = z_target_copy[0]
    Z = z_target_copy[-1]

    if R != Z:
        target_type = 'RL'
        print('Impedance Target is of RL Type')
    else:
        target_type = 'R'
        print('Impedance Target is of R Type')

    return target_type

def R_type_target(z_target, input_z, cap_objs_z, opt, short_prio, check_all = False):

    # This function will be used to generate the proportions/representations of particular capacitor types
    # in the solutions of the initial population


    start_time = time.time()

    cap_list_copy = np.arange(1,np.shape(cap_objs_z)[0]+1, dtype = int)
    total_num_caps = np.shape(cap_list_copy)[0]
    z_copy = np.copy(input_z)
    prio_copy = np.copy(short_prio)
    num_per_cap = np.zeros((total_num_caps), dtype = int)

    #Calculate Group 1 Capacitors
    for i in range(total_num_caps):

        num_per_cap[i] = False  # False meaning that that capacitor type cannot satisfy the last point
        check_all_map = np.zeros((np.shape(z_copy)[1]-1), dtype=int) + i + 1
        all_filled_z = pdn.new_connect_n_decap(z_copy, check_all_map, cap_objs_z, opt)

        if np.abs(all_filled_z)[-1] > z_target[-1] and check_all is False:
            num_per_cap[i] = False # False meaning that that capacitor type cannot satisfy the last point

        else:
            for j in range(1,np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype= int)
                decap_map[prio_copy[0:j]] = i+1

                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)
                if np.abs(some_filled_z)[-1] <= z_target[-1]:
                    num_per_cap[i] = np.count_nonzero(decap_map)
                    break
        print('Required Number of Cap {} to Satisfy Last Freq Pt: {}'.format(i+1, num_per_cap[i] if num_per_cap[i] != 0 else "Cannot Satisfy"))


    ### Edge Case ###
    # Only 1 type of cap can meet final point, only 1 entry will have a non-zero value
    if np.count_nonzero(num_per_cap) == 1:
        caps_group_1 = np.nonzero(num_per_cap)[0] + 1
        num_group_1 = num_per_cap[np.nonzero(num_per_cap)[0]]
        #num_group_1 = np.array([3])

    ### Standard Case:
    else:
        caps_group_1 = np.copy(cap_list_copy)
        num_group_1 = np.copy(num_per_cap)
        for i in range(np.shape(num_group_1)[0]):
            num_group_1[i] = np.shape(z_copy)[1] - 1 if num_group_1[i] == 0 else num_group_1[i]

    print('Caps Group 1 = ', caps_group_1)
    print('Num Group 1 = ', num_group_1)
    print('Time taken for determining decap groups:', time.time() - start_time)

    return caps_group_1, num_group_1


def R_type_target2(z_target, input_z, cap_objs_z, opt, short_prio):

    start_time = time.time()

    cap_list_copy = np.arange(1,np.shape(cap_objs_z)[0]+1, dtype = int)
    total_num_caps = np.shape(cap_list_copy)[0]
    z_copy = np.copy(input_z)
    prio_copy = np.copy(short_prio)
    num_per_cap = np.zeros((total_num_caps), dtype = int)


    #Determine strategy to produce initial population
    temp = np.zeros((total_num_caps), dtype=int)
    for i in range(total_num_caps):
        temp[i] = False
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1
        all_filled_z = pdn.new_connect_n_decap(z_copy, check_all_map, cap_objs_z, opt)

        if np.abs(all_filled_z)[-1] > z_target[-1]:
            temp[i] = False  # False meaning that that capacitor type cannot satisfy the last point

    if np.sum(temp) != 0:
        min_number_pop = True
        print('Strategy 1 Chosen')
    else:
        min_number_pop = False
        print('Strategy 2 Chosen')




    #Calculate Proportions for Group 1 Capacitors
    for i in range(total_num_caps):

        num_per_cap[i] = False  # False meaning that that capacitor type cannot satisfy the last point
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1

        for j in range(1,np.shape(z_copy)[1]):
            decap_map = np.zeros_like(check_all_map, dtype= int)
            decap_map[prio_copy[0:j]] = i+1

            some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)
            if np.abs(some_filled_z)[-1] <= z_target[-1]:
                num_per_cap[i] = np.count_nonzero(decap_map)
                if min_number_pop is True:
                    break
        print('Required Number of Cap {} to Satisfy Last Freq Pt: {}'.format(i+1, num_per_cap[i] if num_per_cap[i] != 0 else "Cannot Satisfy"))


    ### Edge Case ###
    # Only 1 type of cap can meet final point, only 1 entry will have a non-zero value (Only condition we enforce)
    if np.count_nonzero(num_per_cap) == 1:
        caps_group_1 = np.nonzero(num_per_cap)[0] + 1
        num_group_1 = num_per_cap[np.nonzero(num_per_cap)[0]]
    ### Standard Case:
    else:
        caps_group_1 = np.copy(cap_list_copy)
        num_group_1 = np.copy(num_per_cap)
        for i in range(np.shape(num_group_1)[0]):
            num_group_1[i] = np.shape(z_copy)[1] - 1 if num_group_1[i] == 0 else num_group_1[i]

    print('Caps Group 1 = ', caps_group_1)
    print('Num Group 1 = ', num_group_1)
    print('Time taken for determining decap groups:', time.time() - start_time)

    return caps_group_1, num_group_1



def R_type_target3(z_target, input_z, cap_objs_z, opt, short_prio, srf_dir):

    start_time = time.time()
    cap_list_copy = np.arange(1,np.shape(cap_objs_z)[0]+1, dtype = int)
    total_num_caps = np.shape(cap_list_copy)[0]
    z_copy = np.copy(input_z)
    prio_copy = np.copy(short_prio)
    num_per_cap = np.zeros((total_num_caps), dtype = int)

    #Determine strategy to produce initial population
    temp = np.zeros((total_num_caps), dtype=int)


    for i in range(total_num_caps):
        temp[i] = False
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1
        all_filled_z = pdn.new_connect_n_decap(z_copy, check_all_map, cap_objs_z, opt)

        if np.abs(all_filled_z)[-1] <= z_target[-1]:
            temp[i] = 1  # False meaning that that capacitor type cannot satisfy the last point
        else:
            temp[i] = 0
    if np.sum(temp) == np.shape(cap_list_copy)[0]:
        min_number_pop = True
        print('Strategy 1 Chosen')
    else:
        min_number_pop = False
        print('Strategy 2 Chosen')

    #Calculate Proportions for Group 1 Capacitors
    for i in range(total_num_caps):

        num_per_cap[i] = False  # False meaning that that capacitor type cannot satisfy the last point
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1

        unmet = True
        meet_num = None

        if min_number_pop:
            for j in range(1,np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype= int)
                decap_map[prio_copy[0:j]] = i+1

                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)
                if np.abs(some_filled_z)[-1] <= z_target[-1]:
                    num_per_cap[i] = np.count_nonzero(decap_map)
                    break

        else:
            prev_last_z = 0
            for j in range(1,np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype=int)
                decap_map[prio_copy[0:j]] = i + 1
                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)

                if j == 1:
                    prev_last_z = np.abs(some_filled_z[-1])

                if np.abs(some_filled_z)[-1] > prev_last_z:
                    num_per_cap[i] = np.count_nonzero(decap_map)
                    break

                elif np.abs(some_filled_z)[-1] < prev_last_z and\
                        unmet is True and np.abs(some_filled_z)[-1] <= z_target[-1]:
                    meet_num = np.count_nonzero(decap_map)
                    unmet = False
                if j == np.shape(z_copy)[1] - 1:
                    num_per_cap[i] = meet_num if meet_num is not None else False

                prev_last_z = np.abs(some_filled_z[-1])

        print('Required Number of Cap {} to Satisfy Last Freq Pt: {}'.format(i+1, num_per_cap[i] if num_per_cap[i] != 0 else "Cannot Satisfy"))


    ### Edge Case ###
    # Only 1 type of cap can meet final point. generate distribution for the rest of caps
    if np.count_nonzero(num_per_cap) == 1:
        # sole_cap = np.nonzero(num_per_cap)[0] + 1
        # sole_num = num_per_cap[np.nonzero(num_per_cap)[0]]
        # min_num = sole_num[0]
        # max_num = np.shape(z_copy)[1]-1
        # dist = np.linspace(2*min_num,max_num, np.shape(cap_list_copy)[0]-1)
        # dist = np.round(dist)
        # caps_group_1 = np.copy(cap_list_copy)
        # num_group_1 = np.zeros_like(caps_group_1)
        # num_group_1[sole_cap-1] = min_num
        # num_group_1[np.arange(len(caps_group_1), dtype = int)!=sole_cap-1] = dist
        caps_group_1 = np.nonzero(num_per_cap)[0] + 1
        num_group_1 = num_per_cap[np.nonzero(num_per_cap)[0]]
    ### Standard Case:
    else:
        caps_group_1 = np.copy(cap_list_copy)
        num_group_1 = np.copy(num_per_cap)
        for i in range(np.shape(num_group_1)[0]):
            num_group_1[i] = np.shape(z_copy)[1] - 1 if num_group_1[i] == 0 else num_group_1[i]

    print('Caps Group 1 = ', caps_group_1)
    print('Num Group 1 = ', num_group_1)
    print('Time taken for determining decap groups:', time.time() - start_time)

    return caps_group_1, num_group_1



def RL_type_target(z_target, input_z, cap_objs_z, opt, short_prio, check_all = False):

    start_time = time.time()

    cap_list_copy = np.arange(1, np.shape(cap_objs_z)[0] + 1, dtype=int)
    total_num_caps = np.shape(cap_list_copy)[0]
    z_copy = np.copy(input_z)
    prio_copy = np.copy(short_prio)
    num_per_cap = np.zeros((total_num_caps), dtype=int)
    num_per_cap2 = np.zeros((total_num_caps), dtype=int)

    ### Calculate Group 1 Capacitors, the Caps to Satisfy last frequency point
    for i in range(total_num_caps):

        num_per_cap[i] = False
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1
        all_filled_z = pdn.new_connect_n_decap(z_copy, check_all_map, cap_objs_z, opt)
        if np.abs(all_filled_z)[-1] > z_target[-1] and check_all is False:
            num_per_cap[i] = False  # False meaning that that capacitor type cannot satisfy the last point

        else:
            for j in range(1, np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype=int)
                decap_map[prio_copy[0:j]] = i + 1

                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)
                if np.abs(some_filled_z)[-1] <= z_target[-1]:
                    num_per_cap[i] = np.count_nonzero(decap_map)
                    break
        print('Required Number of Cap {} to Satisfy Last Freq Pt: {}'.format(i + 1, num_per_cap[i] if num_per_cap[
                                                                                                          i] != 0 else "Cannot Satisfy"))
    print('\n')
    caps_group_1 = np.copy(cap_list_copy)
    num_group_1 = np.copy(num_per_cap)
    for i in range(np.shape(num_group_1)[0]):
        num_group_1[i] = np.shape(z_copy)[1] - 1 if num_group_1[i] == 0 else num_group_1[i]

    ### Calculate Group 2 Capacitors, the caps to satisfy the transition point ###
    trans_f = np.where((z_target - z_target[0] != 0))[0][0]
    for i in range(total_num_caps):
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1
        all_filled_z = pdn.new_connect_n_decap(z_copy, check_all_map, cap_objs_z, opt)

        if np.abs(all_filled_z)[trans_f] > z_target[trans_f]:
            num_per_cap2[i] = False  # False meaning that that capacitor type cannot satisfy the last point

        else:
            for j in range(1, np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype=int)
                decap_map[prio_copy[0:j]] = i + 1

                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)
                if np.abs(some_filled_z)[trans_f] <= z_target[trans_f]:
                    num_per_cap2[i] = np.count_nonzero(decap_map)
                    break
        print('Required Number of Cap {} to Satisfy Transition Freq Pt: {}'.format(i + 1, num_per_cap2[i] if num_per_cap2[
                                                                                                           i] != 0 else "Cannot Satisfy"))

    caps_group_2 = np.copy(cap_list_copy)
    num_group_2 = np.copy(num_per_cap2)
    for i in range(np.shape(num_group_2)[0]):
        num_group_2[i] = np.shape(z_copy)[1] - 1 if num_group_2[i] == 0 else num_group_2[i]


    ### Edge Case ###
    # Come back to this
    ##########

    caps_group_1 = np.nonzero(caps_group_1)[0] + 1
    caps_group_2 = np.nonzero(caps_group_2)[0] + 1
    caps_group_1 = np.vstack((caps_group_1, caps_group_2))

    num_group_1= num_group_1[np.nonzero(num_group_1)[0]]
    num_group_2 = num_group_2[np.nonzero(num_group_2)[0]]
    num_group_1 = np.vstack((num_group_1, num_group_2))

    print('Caps Group 1 = ', caps_group_1)
    print('Num Group 1 = ', num_group_1)
    print('Time taken for determining decap groups:', time.time() - start_time)

    return caps_group_1, num_group_1




def RL_type_target2(z_target, input_z, cap_objs_z, opt, short_prio, check_all = False):
    #  this function will, later, be used to generate group 1 R_type targets.
    #  codify later to accomodate for the issues.
    #  For now hard code
    start_time = time.time()


    cap_list_copy = np.arange(1, np.shape(cap_objs_z)[0] + 1, dtype=int)
    total_num_caps = np.shape(cap_list_copy)[0]
    z_copy = np.copy(input_z)
    prio_copy = np.copy(short_prio)
    num_per_cap = np.zeros((total_num_caps), dtype=int)
    num_per_cap2 = np.zeros((total_num_caps), dtype=int)

    ######### Calculate Group 1 Capacitors, the Caps to Satisfy last frequency point ########
    temp = np.zeros((total_num_caps), dtype=int)
    for i in range(total_num_caps):
        temp[i] = False
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1
        all_filled_z = pdn.new_connect_n_decap(z_copy, check_all_map, cap_objs_z, opt)

        if np.abs(all_filled_z)[-1] <= z_target[-1]:
            temp[i] = 1  # False meaning that that capacitor type cannot satisfy the last point
        else:
            temp[i] = 0
    if np.sum(temp) == np.shape(cap_list_copy)[0]:
        min_number_pop = True
        print('Strategy 1 Chosen for Last Point')
    else:
        min_number_pop = False
        print('Strategy 2 Chosen for Last Point')



    for i in range(total_num_caps):
        num_per_cap[i] = False  # False meaning that that capacitor type cannot satisfy the last point
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1

        unmet = True
        meet_num = None

        if min_number_pop:
            for j in range(1,np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype= int)
                decap_map[prio_copy[0:j]] = i+1

                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)
                if np.abs(some_filled_z)[-1] <= z_target[-1]:
                    num_per_cap[i] = np.count_nonzero(decap_map)
                    break
        else:
            prev_last_z = 0
            for j in range(1,np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype=int)
                decap_map[prio_copy[0:j]] = i + 1
                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)

                if j == 1:
                    prev_last_z = np.abs(some_filled_z[-1])

                if np.abs(some_filled_z)[-1] > prev_last_z:
                    num_per_cap[i] = np.count_nonzero(decap_map)
                    break

                elif np.abs(some_filled_z)[-1] < prev_last_z and\
                        unmet is True and np.abs(some_filled_z)[-1] <= z_target[-1]:
                    meet_num = np.count_nonzero(decap_map)
                    unmet = False
                if j == np.shape(z_copy)[1] - 1:
                    num_per_cap[i] = meet_num if meet_num is not None else False

                prev_last_z = np.abs(some_filled_z[-1])
        print('Required Number of Cap {} to Satisfy Last Freq Pt: {}'.format(i+1, num_per_cap[i] if num_per_cap[i] != 0 else "Cannot Satisfy"))

    caps_group_1 = np.copy(cap_list_copy)
    num_group_1 = np.copy(num_per_cap)
    for i in range(np.shape(num_group_1)[0]):
        num_group_1[i] = np.shape(z_copy)[1] - 1 if num_group_1[i] == 0 else num_group_1[i]
    #####################################################


    ### Calculate Group 2 Capacitors, the caps to satisfy the transition point ###
    trans_f = np.where((z_target - z_target[0] != 0))[0][0]

    temp = np.zeros((total_num_caps), dtype=int)
    for i in range(total_num_caps):
        temp[i] = False
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1
        all_filled_z = pdn.new_connect_n_decap(z_copy, check_all_map, cap_objs_z, opt)

        if np.abs(all_filled_z)[trans_f] <= z_target[trans_f]:
            temp[i] = 1  # False meaning that that capacitor type cannot satisfy the last point
        else:
            temp[i] = 0

    if np.sum(temp) == np.shape(cap_list_copy)[0]:
        min_number_pop = True
        print('Strategy 1 Chosen for Transition Point')
    else:
        min_number_pop = False
        print('Strategy 2 Chosen for Transition Point')

    for i in range(total_num_caps):

        num_per_cap2[i] = False  # False meaning that that capacitor type cannot satisfy the last point
        check_all_map = np.zeros((np.shape(z_copy)[1] - 1), dtype=int) + i + 1

        unmet = True
        meet_num = None

        if min_number_pop:
            for j in range(1,np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype= int)
                decap_map[prio_copy[0:j]] = i+1

                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)
                if np.abs(some_filled_z)[trans_f] <= z_target[trans_f]:
                    num_per_cap2[i] = np.count_nonzero(decap_map)
                    break
        else:
            prev_last_z = 0
            for j in range(1, np.shape(z_copy)[1]):
                decap_map = np.zeros_like(check_all_map, dtype=int)
                decap_map[prio_copy[0:j]] = i + 1
                some_filled_z = pdn.new_connect_n_decap(z_copy, decap_map, cap_objs_z, opt)

                if j == 1:
                    prev_last_z = np.abs(some_filled_z[trans_f])

                if np.abs(some_filled_z)[trans_f] > prev_last_z:
                    num_per_cap2[i] = np.count_nonzero(decap_map)
                    break

                elif np.abs(some_filled_z)[trans_f] < prev_last_z and \
                        unmet is True and np.abs(some_filled_z)[trans_f] <= z_target[trans_f]:
                    meet_num = np.count_nonzero(decap_map)
                    unmet = False
                if j == np.shape(z_copy)[1] - 1:
                    num_per_cap2[i] = meet_num if meet_num is not None else False

                prev_last_z = np.abs(some_filled_z[trans_f])

        print('Required Number of Cap {} to Satisfy Transition Freq Pt: {}'.format(i + 1, num_per_cap2[i] if num_per_cap2[
                                                                                                           i] != 0 else "Cannot Satisfy"))

    caps_group_2 = np.copy(cap_list_copy)
    num_group_2 = np.copy(num_per_cap2)
    for i in range(np.shape(num_group_2)[0]):
        num_group_2[i] = np.shape(z_copy)[1] - 1 if num_group_2[i] == 0 else num_group_2[i]

    ###### Edge cases were not conidered for the transition point ######
    # those will just be uniform distribution if they all fail #

    #########################################################

    caps_group_1 = np.nonzero(caps_group_1)[0] + 1
    caps_group_2 = np.nonzero(caps_group_2)[0] + 1
    caps_group_1 = np.vstack((caps_group_1, caps_group_2))

    num_group_1 = num_group_1[np.nonzero(num_group_1)[0]]
    num_group_2 = num_group_2[np.nonzero(num_group_2)[0]]
    num_group_1 = np.vstack((num_group_1, num_group_2))

    print('Caps Group 1 = ', caps_group_1)
    print('Num Group 1 = ', num_group_1)
    print('Time taken for determining decap groups:', time.time() - start_time)

    return caps_group_1, num_group_1






def get_probability_vector(input_z, cap_objs_z, opt, short_prio):

    cap_list_copy = np.arange(1, np.shape(cap_objs_z)[0] + 1, dtype=int)
    z_copy = np.copy(input_z)
    prio_copy = np.copy(short_prio)
    p_vector = np.zeros_like(cap_list_copy)
    srf_dir = np.zeros_like(cap_list_copy, dtype= int)

    for i in cap_list_copy:

        map_one = np.zeros(np.shape(input_z)[0] - 1, dtype = int)
        map_one[prio_copy[0]] = i
        z_one = np.abs(pdn.new_connect_n_decap(z_copy, map_one, cap_objs_z, opt))
        one_min = argrelextrema(z_one, np.less)[0] # the last minimum should always be the SRF (discounting funny cavity stuff)

        if one_min.size == 0:
            one_min = np.array([np.shape(input_z)[0]-1], dtype =int)
            srf_dir[i-1] = 1
        elif one_min.size >= 1:
            one_min = np.array([one_min[-1]], dtype =int)
            srf_dir[i-1] = -1

        # map_all = np.zeros_like(map_one)
        # map_all = map_all + i
        #z_all = np.abs(pdn.new_connect_n_decap(z_copy, map_all, cap_objs_z, opt))
        # one_max = argrelextrema(z_all, np.less)[0]
        # if one_max.size == 0:
        #     one_max = np.array([np.shape(input_z)[1]-1], dtype = int)
        # elif one_max.size > 1:
        #     one_max = np.array([one_max[-1]], dtype=int)
        p_vector[np.where(cap_list_copy == i)[0][0]] = np.round(one_min)

    return p_vector, srf_dir

def get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201, interp='log'):
    f_transit = fstop * R / Zmax
    if interp == 'log':
        freq = np.logspace(np.log10(fstart), np.log10(fstop), nf)
    elif interp == 'linear':
        freq = np.linspace(fstart, fstop, nf)
    ztarget_freq = np.array([fstart, f_transit, fstop])
    ztarget_z = np.array([R, R, Zmax])
    ztarget = np.interp(freq, ztarget_freq, ztarget_z)
    return freq, ztarget



########## For shorting vias and port priority ###########

def get_L_mat_from_Z_mat(Z_mat):
    # Z_mat is assumed fxNxN impedance matrix where N are the ports and N > 1 and f are the frequency points
    # this function constructs an L matrix from Z matrix. L matrix will be of shape NxN including self and mutual L
    Zmat_copy = np.copy(Z_mat)
    imag_zmat = np.imag(Zmat_copy)
    L_mat = np.delete(imag_zmat, range(np.shape(Zmat_copy)[0]-1), axis = 0) / (2 * pi * 20e6)
    L_mat = np.reshape(L_mat, (np.shape(Zmat_copy)[1], np.shape(Zmat_copy)[1]))

    return L_mat


def short_all_vias(L_mat):
    # L_mat is an MxM matrix

    # function to determine order the order in which the M ports of the L matrix should be left open

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0])) # get inverse
    shorted_Leq = np.ndarray((1,np.shape(short_copy)[0]-1))
    for i in range(1, np.shape(short_copy)[0]):

        ports_to_short = [j for j in range(1,np.shape(short_copy)[0]) if
                          j != i]  # short every port except the i'th port to see the effect of leaving 1 unshorted
        ports_to_short = [0] + ports_to_short # get ic port

        B_new = B[np.ix_(ports_to_short, ports_to_short)]  # extract out only the rows and columns to short

        # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
        Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
        Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0])) # should end up as a 2x2 always
        shorted_Leq[0,i-1] = Leq[0,0] + Leq[1,1] - Leq[0,1] - Leq[1,0]

    # sorts the Leq in increasing L. This tells you which port, if you left open, would result in the lowest Leq seen
    # looking into port 1
    short_prio = np.argsort(shorted_Leq)

    return short_prio


def short_all_vias2(L_mat):
    # L_mat is an MxM matrix

    # function to determine order the order in which the M ports of the L matrix should be left open
    # not looped to find the best

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0]))  # get inverse
    shorted_Leq = np.ndarray((1, np.shape(short_copy)[0] - 1))
    for i in range(1, np.shape(short_copy)[0]):
        ports_to_short = [i]
        ports_to_short = [0] + ports_to_short  # get IC observation port into L matrix
        B_new = B[np.ix_(ports_to_short, ports_to_short)]
        # extract out only the rows and columns of ports to short and observation port

        # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
        Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
        Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0]))  # should end up as a 2x2 always
        shorted_Leq[0, i - 1] = Leq[0, 0] + Leq[1, 1] - Leq[0, 1] - Leq[1, 0]

    # sorts the Leq in increasing L. This tells you which port, if you short, would give you what equivalent L.
    short_prio = np.argsort(shorted_Leq)

    return short_prio



def loop_open_all_ports(L_mat):
    # L_mat is an MxM matrix

    # function to determine order the order in which the M ports of the L matrix should be left open
    # not looped to find the best

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0]))  # get inverse
    shorted_Leq = np.ndarray((1, np.shape(short_copy)[0] - 1))
    short_prio = np.empty_like(shorted_Leq, dtype= int)
    ports_not_open = list(range(1,np.shape(short_copy)[0]))

    for x in range(1, np.shape(short_copy)[0]):
        ports_open_tracker = []
        temp_Leq = np.ndarray((1, len(ports_not_open) ))
        pos_tracker = 0

        for i in ports_not_open:

            if len(ports_not_open) != 1:
                ports_to_short = [j for j in ports_not_open if j != i]
            else:
                ports_to_short = [j for j in ports_not_open]
            # short every port except ports already calculated as open previously

            ports_to_short = [0] + ports_to_short
            # get ports that should be shorted + IC ports

            B_new = B[np.ix_(ports_to_short, ports_to_short)]
            # extract out only the rows and columns of ports to short and observation port

            Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
            # merge ports by adding the corresponding rows and columns, assuming IC port is index 0

            Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0]))
            # should end up as a 2x2 always

            temp_Leq[0, pos_tracker] = Leq[0, 0] + Leq[1, 1] - Leq[0, 1] - Leq[1, 0]
            ports_open_tracker = ports_open_tracker + [i]
            pos_tracker = pos_tracker + 1

        sorted_temp_Leq = np.argsort(temp_Leq)[0]
        short_prio[0,x-1] = ports_open_tracker[sorted_temp_Leq[-1]]
        # record the port that if not shorted, gives highest Leq (therefore should be shorted)
        ports_not_open.remove(ports_open_tracker[sorted_temp_Leq[-1]])


    short_prio = short_prio - 1
    short_prio = np.flip(short_prio)
    return short_prio



def loop_short_all_ports(L_mat):
    # L_mat is an MxM matrix

    # function to determine order the order in which the M ports of the L matrix should be shorted
    # Not looped for every iteration, just one

    short_copy = np.copy(L_mat)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0]))  # get inverse
    shorted_Leq = np.ndarray((1, np.shape(short_copy)[0] - 1))
    short_prio = np.empty_like(shorted_Leq, dtype= int)
    ports = list(range(1,np.shape(short_copy)[0]))
    ports_already_shorted = []

    for x in range(1, np.shape(short_copy)[0]):

        ports_shorted_tracker = []
        temp_Leq = np.ndarray((1, len(ports)))
        pos_tracker = 0
        #print('Ports:', ports)
        for i in ports:
            ports_to_short = [0] + [i]
            ports_to_short = ports_to_short + [j for j in ports_already_shorted if j not in ports_to_short]
            ports_to_short.sort()
            # get ports that are already shorted + IC ports
            # the i'th port is checking for the next port to short

            B_new = B[np.ix_(ports_to_short, ports_to_short)]
            # extract out only the rows and columns of ports to short and observation port

            Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
            # merge ports by adding the corresponding rows and columns, assuming IC port is index 0

            Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0]))
            # should end up as a 2x2 always

            temp_Leq[0, pos_tracker] = Leq[0, 0] + Leq[1, 1] - Leq[0, 1] - Leq[1, 0]
            ports_shorted_tracker = ports_shorted_tracker + [i]
            pos_tracker = pos_tracker + 1

        sorted_temp_Leq = np.argsort(temp_Leq)[0]
        short_prio[0,x-1] = ports_shorted_tracker[sorted_temp_Leq[0]]
        # record the port that if not shorted, gives highest Leq (therefore should be shorted)
        ports.remove(ports_shorted_tracker[sorted_temp_Leq[0]])

        #print(ports_shorted_tracker)
        # print(sorted_temp_Leq[0])
        ports_already_shorted = ports_already_shorted + [ports_shorted_tracker[sorted_temp_Leq[0]]]

    short_prio = short_prio - 1  #shift from 1 - N  to 0 - N - 1 where N is the total number of ports

    return short_prio

def short_some_ports(L_mat, ports_shorted):

    # This funciton gets you the equivalent inductance looking into IC port when shorting the ports given
    # by ports_shorted

    # L_mat is the L matrix
    # ports_shorted is a list of ports to short
    short_copy = np.copy(L_mat)
    ports_shorted_copy = np.copy(ports_shorted)
    B = np.linalg.solve(short_copy, np.eye(np.shape(short_copy)[0]))  # get inverse

    ports_to_short = [0] + ports_shorted_copy
    ports_to_short.sort()

    B_new = B[np.ix_(ports_to_short, ports_to_short)]
    # extract out only the rows and columns of ports to short and observation port

    Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
    # merge ports by adding the corresponding rows and columns, assuming IC port is index 0

    Leq = np.linalg.solve(Beq, np.eye(np.shape(Beq)[0]))
    # should end up as a 2x2 always

    L_seen = Leq[0, 0] + Leq[1, 1] - Leq[0, 1] - Leq[1, 0]

    return L_seen

