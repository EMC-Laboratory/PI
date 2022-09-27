
'''
Copyright 2020 Ryan (Mohammad) Solgi
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

###############################################################################
###############################################################################
###############################################################################

import numpy as np
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
import copy
import random
from numba import jit
###############################################################################
###############################################################################
###############################################################################

class geneticalgorithm():
    '''  Genetic Algorithm (Elitist version) for Python

    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.


    Implementation and output:

        methods:
                run(): implements the genetic algorithm

        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }

                report: a list including the record of the progress of the
                algorithm over iterations
    '''

    #############################################################
    def __init__(self, function, dimension, caps_group1, num_group1, variable_type='bool',
                 variable_boundaries=None,
                 variable_type_mixed=None,
                 function_timeout=10,
                 algorithm_parameters={'max_num_iteration': None, \
                                       'population_size': 100, \
                                       'mutation_probability': 0.1, \
                                       'elit_ratio': 0.01, \
                                       'crossover_probability': 0.5, \
                                       'parents_portion': 0.3, \
                                       'crossover_type': 'uniform', \
                                       'max_iteration_without_improv': None},
                 seed_sol = None,
                 L_mat = None,
                 record_solutions = False,
                 f_sols = None,
                 short_prio = None):

        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function.
        (For maximization multiply function by a negative sign: the absolute
        value of the output would be the actual objective function)

        @param dimension <integer> - the number of decision variables

        @param variable_type <string> - 'bool' if all variables are Boolean;
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)

        @param variable_boundaries <numpy array/None> - Default None; leave it
        None if variable_type is 'bool'; otherwise provide an array of tuples
        of length two as boundaries for each variable;
        the length of the array must be equal dimension. For example,
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first
        and upper boundary 200 for second variable where dimension is 2.

        @param variable_type_mixed <numpy array/None> - Default None; leave it
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first
        variable is integer but the second one is real the input is:
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1]
        in variable_boundaries. Also if variable_type_mixed is applied,
        variable_boundaries has to be defined.

        @param function_timeout <float> - if the given function does not provide
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function.

        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int>
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of
            successive iterations without improvement. If None it is ineffective

        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm

        '''
        self.__name__ = geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)), "function must be callable"

        self.f = function
        #############################################################
        # dimension

        self.dim = int(dimension)

        # a limiter to control max # of capacitors in a solution
        self.size = 50           # max # of capacitors allowed
        self.size_variation = 5   # allows for solutions in size - size_variation to exist so solutions actually improve

        ######### Some additional constraints to add onto the GA ######################

        # If seeding the GA with some solution
        if seed_sol is not None:
            self.seed_sol = seed_sol.copy()
        else:
            self.seed_sol = None

        # If implementing an L matrix to narrow ports
        if L_mat is not None:
            self.L_mat = np.copy(L_mat)
            default_short_order = self.short_1_via_check(self.L_mat,list(range(1, self.dim + 1)))
            self.ports_to_keep = default_short_order[0:10]
            print(self.ports_to_keep)
        else:
            self.L_mat = None
            self.ports_to_keep = None

        self.zeros_prev = None
        self.initial_sol = False

        # Fille names for writing solutions
        if record_solutions:
            self.f_sols = f_sols
            self.record_sols = True
        else:
            self.f_sols = None
            self.record_sols = False

        if short_prio is not None:
            self.short_prio = short_prio

        # Testing new groups
        self.cap_group1 = np.copy(caps_group1)
        self.num_group1 = np.copy(num_group1)

        #############################################################
        # input variable type

        assert (variable_type == 'bool' or variable_type == 'int' or \
                variable_type == 'real'), \
            "\n variable_type must be 'bool', 'int', or 'real'"
        #############################################################
        # input variables' type (MIXED)

        if variable_type_mixed is None:

            if variable_type == 'real':
                self.var_type = np.array([['real']] * self.dim)
            else:
                self.var_type = np.array([['int']] * self.dim)

        else:
            assert (type(variable_type_mixed).__module__ == 'numpy'), \
                "\n variable_type must be numpy array"
            assert (len(variable_type_mixed) == self.dim), \
                "\n variable_type must have a length equal dimension."

            for i in variable_type_mixed:
                assert (i == 'real' or i == 'int'), \
                    "\n variable_type_mixed is either 'int' or 'real' " + \
                    "ex:['int','real','real']" + \
                    "\n for 'boolean' use 'int' and specify boundary as [0,1]"

            self.var_type = variable_type_mixed
        #############################################################
        # input variables' boundaries

        if variable_type != 'bool' or type(variable_type_mixed).__module__ == 'numpy':

            assert (type(variable_boundaries).__module__ == 'numpy'), \
                "\n variable_boundaries must be numpy array"

            assert (len(variable_boundaries) == self.dim), \
                "\n variable_boundaries must have a length equal dimension"

            for i in variable_boundaries:
                assert (len(i) == 2), \
                    "\n boundary for each variable must be a tuple of length two."
                assert (i[0] <= i[1]), \
                    "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound = variable_boundaries
        else:
            self.var_bound = np.array([[0, 1]] * self.dim)

        #############################################################
        # Timeout
        self.funtimeout = float(function_timeout)

        #############################################################
        # input algorithm's parameters

        self.param = algorithm_parameters

        self.pop_s = int(self.param['population_size'])

        assert (self.param['parents_portion'] <= 1 \
                and self.param['parents_portion'] >= 0), \
            "parents_portion must be in range [0,1]"

        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param['mutation_probability']

        assert (self.prob_mut <= 1 and self.prob_mut >= 0), \
            "mutation_probability must be in range [0,1]"

        self.prob_cross = self.param['crossover_probability']
        assert (self.prob_cross <= 1 and self.prob_cross >= 0), \
            "mutation_probability must be in range [0,1]"

        assert (self.param['elit_ratio'] <= 1 and self.param['elit_ratio'] >= 0), \
            "elit_ratio must be in range [0,1]"

        trl = self.pop_s * self.param['elit_ratio']
        if trl < 1 and self.param['elit_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert (self.par_s >= self.num_elit), \
            "\n number of parents must be greater than number of elits"

        if self.param['max_num_iteration'] == None:
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == 'int':
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * self.dim * (100 / self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * 50 * (100 / self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(self.param['max_num_iteration'])

        self.c_type = self.param['crossover_type']
        assert (self.c_type == 'uniform' or self.c_type == 'one_point' or \
                self.c_type == 'two_point'), \
            "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

        self.stop_mniwi = False
        if self.param['max_iteration_without_improv'] == None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])

        #############################################################
    def run(self):

        #############################################################

        # Initial Population
        self.integers = np.where(self.var_type == 'int')
        self.reals = np.where(self.var_type == 'real')

        #initialize population
        pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)

        # stores each solutio
        solo = np.zeros(self.dim + 1)

        #Each var is a solution
        var = np.zeros(self.dim)

        # used for seeding initial solutions ('deleteable')
        if self.seed_sol is not None:
            # if using a seed solution, update parameters
            print("Initial Solution Accepted")
            print("Initial Solution is", self.seed_sol)

            obj = self.sim(self.seed_sol)
            self.best_function = obj
            self.best_variable = self.seed_sol.copy()
            self.size = np.count_nonzero(self.best_variable)

            print("Score given as", self.best_function)
            print("# of Capacitors Per Solution Starting at", self.size)

        ## Generate Population with NN Inputs
        self.size = 35
        self.size_variation = 2
        #pop = self.percentage_pop3(0.8, 21, pop, self.size_variation, short_prio= self.short_prio)
        #pop = self.percentage_pop_ran_caps(0.625, int(self.size - self.size_variation), pop, self.size_variation, short_prio=self.short_prio)
        #pop = self.percentage_pop_small_caps(0.54, int(self.size - self.size_variation), pop, self.size_variation, short_prio=self.short_prio)
        #pop = self.percentage_pop_small_caps_test(0.54, int(self.size - self.size_variation), pop, self.size_variation, short_prio=self.short_prio)
        #pop = self.roulette_pop(0.54, int(self.size - self.size_variation), pop, self.size_variation, short_prio=self.short_prio)

        ################## TEST NEW STUFF
        #pop =  self.R_target_group(int(self.size - self.size_variation), pop, self.size_variation, self.cap_group1, self.num_group1, short_prio=self.short_prio)
        pop =  self.RL_target_group(int(self.size - self.size_variation), pop, self.size_variation, self.cap_group1, self.num_group1, short_prio=self.short_prio)

        for i in range(self.pop_s):
            pop[i][0:-1] = self.shift_mut(self.short_prio, pop[i][0:-1], .1, move_max=10)
            #print('Sol:', pop[i][0:-1])
            obj = self.sim(np.copy(pop[i][0:-1]))
            pop[i][-1] = obj
            #if pop[i][-1] < 0:
            #    print('Solution in Initial Pop:', pop[i][0:-1], 'with', np.count_nonzero(pop[i][0:-1]), 'caps')

        print('Num Sols in Initial Pop = ', np.count_nonzero(np.where(pop[:,-1] < 0)[0]))
        ###################


        ## randomly generate population with uniform distribution #####
        # for p in range(0, self.pop_s):
        #
        #     if p == 0 and self.seed_sol is not None:
        #         var = self.seed_sol.copy()
        #
        #     else:
        #         for i in self.integers[0]:
        #             #var[i] = np.random.randint(self.var_bound[i][0], \
        #             #                          self.var_bound[i][1] + 1)
        #             #solo[i] = var[i].copy()
        #             var[i] = np.random.randint(1,10)
        #         for i in self.reals[0]:
        #             var[i] = self.var_bound[i][0] + np.random.random() * \
        #                      (self.var_bound[i][1] - self.var_bound[i][0])
        #             #solo[i] = var[i].copy()
        #
        #     ###### Initial population check for pop size -- Jack Jank ######
        #     n_caps = np.count_nonzero(var)
        #     if n_caps < (self.size - self.size_variation) or n_caps > self.size or n_caps == 0:
        #         var = self.pop_size_control(var, self.size, random_change=False, num_cap_mod= random.randint(0,self.size_variation+1))
        #         #var = self.pop_size_control_w_shorts(var, self.size, random_change=False, ports_to_keep=self.ports_to_keep,num_cap_mod= random.randint(0,self.size_variation))
        #     for i in range(np.shape(var)[0]):
        #         solo[i] = var[i].copy()
        #     obj = self.sim(var)
        #     solo[self.dim] = obj
        #     pop[p] = solo.copy()



        #############################################################


        #############################################################
        # Report
        obj = 1000000
        # I have to reset obj here, otherwise if I don't, obj will just be the score of the last member of the population
        # This breaks the GA in the odd case where if the last solution meets the target impedance, and it has
        # n caps = size - size_variation, and then a better solution than that is never found,
        # that solution won't be recorded
        self.report = []
        self.test_obj = obj
        self.best_variable = var.copy()
        self.best_function = obj
        ##############################################################

        t = 1
        counter = 0
        while t <= self.iterate:
            s1 = time.time()
            self.progress(t, self.iterate, status="GA is running...")
            #############################################################
            # Sort
            #print('\n', np.argmin(pop[:,-1]))
            #print(pop[np.argmin(pop[:,-1])][-1])

            pop = pop[pop[:, self.dim].argsort()]

            #############################
            if pop[0, self.dim] < self.best_function:

                # if best scoring member of the population is better than current best score,
                # overwrite the best score
                # overwrite the previous best performing solution with the new best solution
                counter = 0
                self.best_variable = pop[0, : self.dim].copy()
                self.best_function = self.f(self.best_variable)

                if self.best_function < 0:
                    print('\nTotal Number of Capacitors has Decreased. Minimum Decap Number =', np.count_nonzero(self.best_variable))
                    print('Best Solution is:', self.best_variable, '\n')

                    prev_sol = np.copy(self.best_variable)
                    self.best_variable = self.sweep_check4(self.best_variable, self.short_prio)
                    self.size = np.count_nonzero(self.best_variable)

                    print('\nAfter Performing Sweep Check, Minimum Decap Number =', np.count_nonzero(self.best_variable))
                    print('Best Solution is:', self.best_variable)
                    print('Size Parameter = ', self.size)

                    # adjust decap number to account for sweep check reducton
                    if np.count_nonzero(self.best_variable) < np.count_nonzero(prev_sol): #sweep check activated
                        for i in pop:
                            n_caps = np.count_nonzero(i[0:self.dim])
                            if n_caps < (self.size - self.size_variation) or n_caps > self.size or n_caps == 0:
                                new_sol = self.pop_size_control(i[0:self.dim], self.size, random_change=False,
                                                                num_cap_mod=random.randint(0, self.size_variation-1) + 1)
                                obj = self.sim(new_sol)
                                i[0:self.dim] = np.copy(new_sol)
                                i[self.dim] = obj


                    ######################################
                    self.best_function = self.f(self.best_variable)
                    pop[0,self.dim] = self.f(self.best_variable)
                    pop[0,:self.dim] = np.copy(self.best_variable)


                else:
                    self.best_function = pop[0, self.dim].copy()
                    self.best_variable = pop[0, : self.dim].copy()

                    # update the largest possible size of the solution
                    # based on # of decaps of the best soluiton, if target is met
                print('\nScore Improved. Best Score So Far Is', self.best_function,
                      '\nNum Caps:', self.dim + self.best_function + 1 if self.best_function < 0 else
                      'Target Not Met ({} Caps)'.format(np.count_nonzero(self.best_variable)))
                print('Solution is', self.best_variable)
            else:
                counter += 1

            if self.record_sols:
                sol_to_list = [int(i) for i in self.best_variable.tolist()]
                sol_to_list = str(sol_to_list)
                self.f_sols.write(sol_to_list)
                self.f_sols.write('\n')

            ##############
            # if pop[0, self.dim] < self.best_function:
            #
            #     # if best scoring member of the population is better than current best score,
            #     # overwrite the best score
            #     # overwrite the best performing member with the new best member
            #     counter = 0
            #
            #     self.best_function = pop[0, self.dim].copy()
            #     self.best_variable = pop[0, : self.dim].copy()
            #
            #     # update the largest possible size of the solution
            #     # based on # of decaps of the best soluiton, if target is met
            #     print('Score Improved. Best Score So Far Is', self.best_function)
            #
            #     if self.best_function < 0:
            #
            #         # if self.L_mat is not None:
            #         #     self.best_variable = self.sweep_check3(self.best_variable)
            #         # else:
            #         #     self.best_variable = self.sweep_check(self.best_variable)
            #         self.best_function = self.f(self.best_variable)
            #         pop[0,self.dim] = self.f(self.best_variable)
            #         pop[0,:self.dim] = np.copy(self.best_variable)
            #
            #     if np.count_nonzero(self.best_variable) < self.size and self.best_function < 0:
            #         self.size = np.count_nonzero(self.best_variable)
            #         print('Total Number of Capacitors has Decreased. Minimum Decap Number =', self.size)
            #
            #     print('Solution is', self.best_variable)
            # else:
            #     counter += 1


            # Report
            self.report.append(pop[0, self.dim])  # the best score of each generation


            ########## Checking something ############
            #
            #for i in pop:
                #if i[-1] < 0:
            #    print(np.count_nonzero(i[0:-1]))
            if self.best_function < 0:
                sols_meet_target = np.where(pop[:,-1] < 0)[0]
                #print('Unique Sols:', np.unique(pop[np.s_[sols_meet_target], 0:-1], axis = 0))
                print('Num Unique =', np.shape(np.unique(pop[np.s_[sols_meet_target], 0:-1], axis = 0))[0])

            ##############################################################
            # Normalizing objective function

            normobj = np.zeros(self.pop_s)

            minobj = pop[0, self.dim]
            if minobj < 0: # would be < 0 if your goal is to maximize (check the objective function section)
                normobj = pop[:, self.dim] + abs(minobj)

            else:
                normobj = pop[:, self.dim].copy()

            maxnorm = np.amax(normobj)
            normobj = maxnorm - normobj + 1  # the smallest value would still be the smallest

            #############################################################
            # Calculate probability

            sum_normobj = np.sum(normobj)

            prob = np.zeros(self.pop_s)
            prob = normobj / sum_normobj
            cumprob = np.cumsum(prob)

            #############################################################
            # Select parents
            par = np.array([np.zeros(self.dim + 1)] * self.par_s)

            for k in range(0, self.num_elit):
                # copy some elite members over. num of elite members determined by elite ratio
                par[k] = pop[k].copy()
            for k in range(self.num_elit, self.par_s):


                index = np.searchsorted(cumprob, np.random.random())
                par[k] = pop[index].copy()


            ef_par_list = np.array([False] * self.par_s)
            par_count = 0

            # Basically a loop to decide which of the selected parents will actually crossover
            # Denoted as a True/False array
            while par_count == 0:
                for k in range(0, self.par_s):
                    if np.random.random() <= self.prob_cross:
                        ef_par_list[k] = True
                        par_count += 1

            # If i am not mistaken, par[ef_par_list] will return an array comprising of only those indices
            # where ef_par_list is true.
            # ie par[ef_par_list] will include only the selected parents (with index = true) that will cross over
            # ef_par is a list of parents that will actually breed
            # ELITE parents are also included here
            # ef_par is a list of parents that CAN actually breed, not a list of all parents
            ef_par = par[ef_par_list].copy()

            #if t == 50:
            #    print('here', ef_par)


            #############################################################
            # New generation
            pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)


            for k in range(0, self.par_s):
                if k <= self.num_elit:
                    pop[k] = par[k].copy()
                else:
                    chance = random.random()
                    if chance > 2: # this doesn't do anything as random.random is in the range of [0,1)

                        new_sol = self.shift_mut(self.short_prio, par[k][0:-1], self.prob_mut, move_max = 3)
                        obj = self.sim(new_sol)
                        pop[k][0:-1] = np.copy(new_sol)
                        pop[k][-1] = obj
                    else:

                        pop[k] = par[k].copy()
            for k in range(self.par_s, self.pop_s, 2):

                r1 = np.random.randint(0, par_count)   # from the parents chosen for breeding, grab one
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1, : self.dim].copy()  # pretty sure it copies the variables of the parents at r1. Excludes the score
                pvar2 = ef_par[r2, : self.dim].copy()  # same here for second parent


                # #Children from crossover

                chance = random.random()

                if chance < -1:
                    ch = self.cross(pvar1, pvar2, self.c_type)
                    ch1 = ch[0].copy()
                    ch2 = ch[1].copy()

                else:
                    ch1 = pvar1.copy()
                    ch2 = pvar2.copy()
                    ch1 = self.cust_mut(ch1, self.prob_mut)
                    ch2 = self.shift_mut(self.short_prio, ch2, self.prob_mut, move_max= 5)

                #ch1 = self.mut(ch1)                     # mutate one of the child 1 in the classical way
                #ch2 = self.mutmidle(ch2, pvar1, pvar2)  # mutates child 2 in a 'fancy' way
                                                        # tries to keep genes mutated within a range based on parents

                # population control for GA
                # ensure only some # of caps are in a solution
                if self.size != 0:
                    n_caps1 = np.count_nonzero(ch1)
                    n_caps2 = np.count_nonzero(ch2)
                    if n_caps1 < (self.size - self.size_variation) or n_caps1 > self.size or n_caps1 == 0:
                        ch1 = self.pop_size_control(ch1, self.size, random_change=False, num_cap_mod = random.randint(0,self.size_variation-1)+1)
                        #random.randint(0,self.size_variation-1)
                        #ch1 = self.pop_size_control_w_shorts(var, self.size, random_change=False, ports_to_keep=self.ports_to_keep,
                        #                                     num_cap_mod=1)
                        #print('a1', np.nonzero(ch1))
                    if n_caps2 < (self.size - self.size_variation) or n_caps2 > self.size or n_caps2 == 0:
                        pass
                        ch2 = self.pop_size_control(ch2, self.size, random_change= False, num_cap_mod= random.randint(0,self.size_variation-1)+1)
                        #random.randint(0,self.size_variation-1)
                        #ch2 = self.pop_size_control_w_shorts(var, self.size, random_change=False, ports_to_keep=self.ports_to_keep,
                        #                                    num_cap_mod=1)
                        #print('n caps 2', np.count_nonzero(ch2))

                solo[: self.dim] = ch1.copy()   # copy the genes over to solo
                obj = self.sim(ch1)             # calculate score/check if score calculatable
                solo[self.dim] = obj            # store score
                pop[k] = solo.copy()            # copy member of population
                solo[: self.dim] = ch2.copy()   # do the same for the second child
                obj = self.sim(ch2)
                solo[self.dim] = obj
                pop[k + 1] = solo.copy()

            print("Current Population:" , t, "\n")
            #for i in pop:
            #    print(np.count_nonzero(i[0:self.dim]))

            t += 1
            print(time.time() - s1)
            # if score does not improve within some # of generations
            if counter > self.mniwi:
                pop = pop[pop[:, self.dim].argsort()]
                if pop[0, self.dim] >= self.best_function:
                    t = self.iterate
                    self.progress(t, self.iterate, status="GA is running...")
                    time.sleep(2)
                    t += 1
                    self.stop_mniwi = True

        #############################################################
        # Sort
        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:
            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, : self.dim].copy()

        if self.record_sols:
            sol_to_list = [int(i) for i in self.best_variable.tolist()]
            sol_to_list = str(sol_to_list)
            self.f_sols.write(sol_to_list)
            self.f_sols.write('\n')

        #############################################################
        # Report

        self.report.append(pop[0, self.dim])

        self.output_dict = {'variable': self.best_variable, 'function': \
            self.best_function}
        show = ' ' * 100
        sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush()
        re = np.array(self.report)

        # plt.plot(re)
        # plt.xlabel('Iteration')
        # plt.ylabel('Objective function')
        # plt.title('Genetic Algorithm')
        # plt.show()
        if self.stop_mniwi == True:
            sys.stdout.write('\nWarning: GA is terminated due to the' + \
                             ' maximum number of iterations without improvement was met!')

    ##############################################################################
    ##############################################################################
    def cross(self, x, y, c_type):

        ofs1 = x.copy()
        ofs2 = y.copy()

        if c_type == 'one_point':
            ran = np.random.randint(0, self.dim)
            for i in range(0, ran):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

                # picks cross over point
                # copies makes two children first, with the single point crossover applied to each

        if c_type == 'two_point':

            ran1 = np.random.randint(0, self.dim)
            ran2 = np.random.randint(ran1, self.dim)

            for i in range(ran1, ran2):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

                # picks cross over points
                # copies makes two children first, with the two point crossover applied to each

        if c_type == 'uniform':

            for i in range(0, self.dim):
                ran = np.random.random()
                if ran < 0.5:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

                    # Uniformly picks crossover points
                    # copies makes two children first, with the uniform crossover applied to each

        return np.array([ofs1, ofs2])

    ###############################################################################

    def mut(self, x):

        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = np.random.randint(self.var_bound[i][0], \
                                         self.var_bound[i][1] + 1)

        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = self.var_bound[i][0] + np.random.random() * \
                       (self.var_bound[i][1] - self.var_bound[i][0])

        return x

    ###############################################################################
    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = np.random.randint(p1[i], p2[i])
                elif p1[i] > p2[i]:
                    x[i] = np.random.randint(p2[i], p1[i])
                else:
                    x[i] = np.random.randint(self.var_bound[i][0], \
                                             self.var_bound[i][1] + 1)

        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = p1[i] + np.random.random() * (p2[i] - p1[i])
                elif p1[i] > p2[i]:
                    x[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
                else:
                    x[i] = self.var_bound[i][0] + np.random.random() * \
                           (self.var_bound[i][1] - self.var_bound[i][0])
        return x

    ###############################################################################
    def evaluate(self):
        return self.f(self.temp)

    ###############################################################################
    def sim(self, X):
        self.temp = X.copy()
        obj = None

        ## How func_timeout works
        # def func_timeout(timeout, func, args=(), kwargs=None
        # func_timeout runs function 'func' with input arguments 'args' for up to 'timeout' seconds
        # At the end of 'timeout' seconds,
        # it will return/raise anything the passed function would otherwise return or raise
        # if 'timeout' is exceeded before any other return/raise from the function would occur (ie function not done)
        # it will return a time out error (FunctionTimedOut)
        try:
            obj = func_timeout(self.funtimeout, self.evaluate)
            # in this case, timeout is funtimeout
            # the function to be evaluated is the input function generally (objective function, self.f, contextually)
            # where the input argument for the input function is self.temp (a copy of something passed to sim)

            # so it seems like funtimeout is a max limit on how long the objective function is allowed to run

            #if it does not time out, and its evaluateable, it returns the scores
        except FunctionTimedOut:
            print("given function is not applicable")
            # is time is exceeded, the objective function (or just input function) is not appropriate for use?

        assert (obj != None), "After " + str(self.funtimeout) + " seconds delay " + \
                              "func_timeout: the given function does not provide any output"
            # assert:, check if a condition is true, if not, raise an error
            # if the function successfully runs all the way through but does not have a return argument, lets you
            # know you need a return argument.

        # So basically it seems like this function checks if the input argument to the input function won't raise errors
        # Checks if the input function will run in a reasonable amount of time
        # checks if the input function actually has a return argument
        return obj

    ###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()
    ###############################################################################

###############################################################################

    ################### Functions Jack Jank Added ###############

    def pop_size_control(self, sol, size, random_change=False, force_empty=None, num_cap_mod=0):
        # this function will take a solution, sol, and changes the number of capacitors for solutions in the population
        # to a specific number


        # for the PDN capacitor selection, a 0 represents an empty port.
        # we want to control max # of capacitors (non-zero) can be in the solution
        # Size is the max # of capacitors allowed in the solution

        # later may change so size is a range but not rn
        # Also to note, from default code, pop has dimension [0:self.dim+1]
        # 0 to self.dim contains the genes
        # the self.dim + 1 should be the evaluation/score

        # pop is the population object
        # size is an integer from 1 to self.dim

        # There might be issues with num_cap_mod causing you to have < 0 caps. I think anyways.
        # not an issue if num_cap_mod = 0
        if not isinstance(size, int):
            raise TypeError('The "Size" of the Solution Must be an Integer In [1, #of Genes]')

        if size < 1 or size > self.dim:
            raise ValueError('Size of a Solution Must Be Between 1 or the # of Genes (# of Ports')

        new_chrome = sol[0:self.dim].copy()  # copies the genes

        if force_empty is not None:  # if there are ports we need to force empty
            for j in force_empty:
                new_chrome[j] = 0

        current_size = np.count_nonzero(new_chrome)  # number of caps in current solution
        caps_locations = np.nonzero(new_chrome)[0]  # gets locations where ports are not empty
        caps = new_chrome[caps_locations]  # gets capacitors at non-empty ports


        if current_size == 0:
            empty_locations = np.nonzero(new_chrome == 0)[0]  # get port locations that have no decaps
            if force_empty is not None:  # if there are ports we need to keep empty
                force_empty_rev = np.flip(force_empty)  # get indices of ports that must be forced empty
                for j in force_empty_rev:
                    empty_locations = np.delete(empty_locations,
                                                np.where(empty_locations == j)[0])  # remove them from the fill list

            num_to_add = np.random.randint(1, size)
            ports_to_fill = np.random.choice(empty_locations, size=num_to_add, replace=False)

            for j in ports_to_fill:
                new_cap = np.random.randint(np.min(self.var_bound[0]) + 1, np.max(self.var_bound[0]) + 1)
                new_chrome[j] = new_cap

        elif current_size > size:
            num_to_empty = current_size - size + num_cap_mod
            if current_size - num_to_empty < size - self.size_variation:
                num_to_empty = current_size - size

            if num_to_empty > current_size:
                num_to_empty = 1
            ports_to_empty = np.random.choice(caps_locations, size=num_to_empty, replace=False)

            for j in ports_to_empty:
                new_chrome[j] = 0


        elif current_size < size - self.size_variation:
            empty_locations = np.nonzero(new_chrome == 0)[0]  # get port locations that have no decaps
            if force_empty is not None:  # if there are ports we need to keep empty
                force_empty_rev = np.flip(force_empty)  # get indices of ports that must be forced empty
                for j in force_empty_rev:
                    empty_locations = np.delete(empty_locations,
                                                np.where(empty_locations == j)[0])  # remove them from the fill list

            num_to_add = size - current_size + num_cap_mod
            if current_size + num_to_add >= size:
                num_to_add = size - current_size

            ports_to_fill = np.random.choice(empty_locations, size=num_to_add, replace=False)
            if random_change:
                for j in ports_to_fill:
                    new_cap = np.random.randint(np.min(self.var_bound[0]) + 1, np.max(self.var_bound[0]) + 1)
                    new_chrome[j] = new_cap
            else:
                # replace with a cap already in solution rather than randomly out of the entire range
                for j in ports_to_fill:
                    new_cap = caps[np.random.randint(0, current_size)]
                    new_chrome[j] = new_cap

        return new_chrome

    def pop_size_control_w_shorts(self, pop, size, random_change=False, ports_to_keep = None, num_cap_mod=0):
        # this function will take a population, pop, and changes the number of capacitors for solutions in the population
        # to a specific number. Will not open up particular ports

        # for the PDN capacitor selection, a 0 represents an empty port.
        # we want to control how many capacitors (non-zero) can be in the solution
        # there will be 'size' number of capacitors

        # later may change so size is a range but not rn

        # Also to note, from default code, pop has dimension [0:self.dim+1]
        # 0 to self.dim contains the genes
        # the self.dim + 1 should be the evaluation/score

        # pop is the population object
        # size is an integer from 1 to self.dim

        if not isinstance(size, int):
            raise TypeError('The "Size" of the Solution Must be an Integer In [1, #of Genes]')

        if size < 1 or size > self.dim:
            raise ValueError('Size of a Solution Must Be Between 1 or the # of Genes (# of Ports')

        # for the edge case where the # of ports to keep is larger than the max allowed # of capacitors.
        # adjusting for decap # won't work.
        # need to fix later.
        if np.shape(ports_to_keep)[0] >= self.size - num_cap_mod and ports_to_keep is not None:
            ports_to_keep = None

        new_pop = pop.copy()
        if np.ndim(new_pop) == 1:  # if you are passing only one solution and not the entire population
            new_pop = np.array([new_pop])

        for i in range(np.shape(new_pop)[0]):
            new_chrome = new_pop[i][0:self.dim].copy()  # copies the genes

            current_size = np.count_nonzero(new_chrome)  # number of caps in solutions
            caps_locations = np.nonzero(new_chrome)[0]  # gets locations where ports are not empty
            caps = new_chrome[caps_locations]  # gets capacitors at non-empty ports

            if ports_to_keep is None:
                if current_size > size:
                    num_to_empty = current_size - size + num_cap_mod
                    ports_to_empty = np.random.choice(caps_locations, size=num_to_empty, replace=False)
                    for j in ports_to_empty:
                        new_chrome[j] = 0
                    new_pop[i][0:self.dim] = new_chrome.copy()

                elif current_size < size:
                    empty_locations = np.nonzero(new_chrome == 0)[0]  # gets empty port locations

                    num_to_add = size - current_size - num_cap_mod
                    ports_to_fill = np.random.choice(empty_locations, size=num_to_add, replace=False)

                    if random_change:
                        for j in ports_to_fill:
                            new_cap = np.random.randint(np.min(self.var_bound[0]) + 1, np.max(self.var_bound[0]) + 1)
                            new_chrome[j] = new_cap
                        # new_pop[i][0:self.dim] = new_chrome.copy()
                    else:
                        # replace with a cap already in solution rather than randomly out of the entire range
                        for j in ports_to_fill:
                            # might consider changing this to make the caps list unique, ie the # of a particular
                            # type isn't weighted to the # of apperances in the solution.
                            new_cap = caps[np.random.randint(0, current_size)]
                            new_chrome[j] = new_cap

                        new_pop[i][0:self.dim] = new_chrome.copy()

            else:
                for j in ports_to_keep:
                    if new_chrome[j] == 0:
                        caps = np.unique(caps)
                        new_cap = caps[np.random.randint(0, caps.shape[0])]
                        new_chrome[j] = new_cap
                        current_size += 1

                if current_size > size:
                    num_to_empty = current_size - size + num_cap_mod

                    # if # of ports you need to remove is less than the # of ports you are allowed to remove from,
                    # then don't remove from the ports that you want to keep decaps in.
                    if num_to_empty < self.size - len(ports_to_keep):
                        holder = np.in1d(caps_locations, ports_to_keep)
                        caps_locations = caps_locations[np.where(holder == 0)]

                    ports_to_empty = np.random.choice(caps_locations, size=num_to_empty, replace=False)
                    for j in ports_to_empty:
                        new_chrome[j] = 0
                    new_pop[i][0:self.dim] = new_chrome.copy()

                elif current_size < size:
                    empty_locations = np.nonzero(new_chrome == 0)[0]  # gets empty port locations
                    num_to_add = size - current_size - num_cap_mod if size - current_size - num_cap_mod > 0 else 0
                    ports_to_fill = np.random.choice(empty_locations, size=num_to_add, replace=False)
                    caps_locations = np.nonzero(new_chrome)[0]  # gets locations where ports are not empty
                    caps = new_chrome[caps_locations]  # gets capacitors at non-empty ports
                    caps = np.unique(caps)
                    if random_change:
                        for j in ports_to_fill:
                            new_cap = np.random.randint(np.min(self.var_bound[0]) + 1, np.max(self.var_bound[0]) + 1)
                            new_chrome[j] = new_cap
                        # new_pop[i][0:self.dim] = new_chrome.copy()
                    else:
                        # replace with a cap already in solution rather than randomly out of the entire range
                        for j in ports_to_fill:
                            new_cap = caps[np.random.randint(0, caps.shape[0])]
                            new_chrome[j] = new_cap
                        new_pop[i][0:self.dim] = new_chrome.copy()

        if np.shape(new_pop)[0] == 1:
            new_pop = new_pop[0]

        return new_pop

    def shift_mut(self, short_prio, sol, mut_rate, move_max = 3):
        short_copy = np.copy(short_prio)
        new_sol = np.copy(sol)
        for i in range(np.shape(sol)[0]):
            chance = random.random()
            if chance < mut_rate:
                move = random.randint(1, move_max)
                loc_ind = np.where(short_copy == i)[0]
                if loc_ind != short_copy[0] or loc_ind != short_copy[-1]:
                    if i == 0:
                        direction = 1
                    elif i == np.shape(sol)[0]:
                        direction = 0
                    else:
                        direction = random.randint(0, 1)  # 0 means moving towards ports that should be empty
                                                      # 1 means moving towards ports that shouldn't be empty
                    if direction == 1:
                        new_loc = loc_ind + move if loc_ind + move < np.shape(short_copy)[0] - 1 else \
                        np.shape(short_copy)[0] - 1
                        if new_sol[new_loc] == 0:
                            new_sol[new_loc] = new_sol[loc_ind]
                            new_sol[loc_ind] = 0
                        else:
                            old_value = new_sol[loc_ind]
                            new_sol[loc_ind] = new_sol[new_loc]
                            new_sol[new_loc] = old_value
                    else:
                        new_loc = loc_ind - move if loc_ind - move > 0 else 0
                        if new_sol[new_loc] == 0:
                            new_sol[new_loc] = new_sol[loc_ind]
                            new_sol[loc_ind] = 0
                        else:
                            old_value = new_sol[loc_ind]
                            new_sol[loc_ind] = new_sol[new_loc]
                            new_sol[new_loc] = old_value
        return new_sol

    def cust_mut(self, sol, mut_rate, mut_zeros = True):
        new_sol = np.copy(sol)
        cap_list = np.arange(self.var_bound[0][1]) + 1
        for i in range(np.shape(new_sol)[0]):
            chance = np.random.random()
            if chance < mut_rate:
                old_cap = new_sol[i]
                pos = np.where(cap_list == old_cap)[0]

                if pos != 0 and pos != (np.shape(cap_list)[0] - 1):
                    new_cap = [cap_list[pos - 1], cap_list[pos + 1]]
                    ran = random.random()
                    if ran < .5:
                        new_cap = new_cap[0]
                    else:
                        new_cap = new_cap[1]
                elif old_cap == 0:
                    if mut_zeros:
                        new_cap = np.random.choice(cap_list)
                    else:
                        new_cap = 0
                elif pos == 0:
                    new_cap = cap_list[pos + 1]
                else:
                    new_cap = cap_list[pos - 1]
                new_sol[i] = new_cap

        return new_sol

    def reduce_via_short(self, mergedL, ports_shorted, ic_port=0):

        # merged L is the L matrix with IC vias merged. IC via included in array and assumed as port 0
        # ports_shorted is a list of already shorted ports (ports with decaps already placed)

        # Currently does not work if there are 2 ports left and you are deciding which via to remove next.

        B = np.linalg.inv(mergedL)  # get B matrix
        Leq_mat = np.zeros(len(ports_shorted))  # holder for storing equivalent inductances
        short_prio = np.array(ports_shorted)
        for i in range(len(ports_shorted)):
            ports_to_short = [j for j in ports_shorted if
                              j != ports_shorted[i]]  # short every port except the i'th port. Short 1 port at a time
            B_new = B[np.ix_(ports_to_short, ports_to_short)]  # extract out only the rows and columns to short
            # merge ports by adding the corresponding rows and columns, assuming IC port is index 0
            Beq = np.add.reduceat(np.add.reduceat(B_new, [0, 1], axis=0), [0, 1], axis=1)
            L = np.linalg.inv(Beq)
            Leq = L[0, 0] + L[1, 1] - L[0, 1] - L[1, 0]
            Leq_mat[i] = Leq
        Leq_sorted = np.argsort(Leq_mat)  # sort inductances from lowest to highest
        short_prio = (short_prio[np.s_[Leq_sorted]])  # relate L to the port number, still lowest to highest
        return short_prio

    def short_1_via_check(self, L, ports_shorted, ic_port=0):
        # functions to mass short only 1 via at a time
        Leq_mat = np.zeros(len(ports_shorted))  # holder for storing equivalent inductances
        short_prio = np.array(ports_shorted)
        for i in range(len(ports_shorted)):
            # 'i' is the current port to short
            port = ports_shorted[i]
            Leq = L[0, 0] + L[port, port] - L[0, port] - L[port, 0]
            Leq_mat[i] = Leq
        Leq_sorted = np.argsort(Leq_mat)  # sort inductances from lowest to highest
        short_prio = (short_prio[np.s_[Leq_sorted]])  # relate the sorted L to the port number
        return short_prio

    def sweep_check(self, min_zero_map): # should use this at some point
        improve_bool = True
        stop_ind = 0
        while improve_bool is True:
            current_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # number of caps in the initial min_zero_map
            print('Before check, decap map is:', min_zero_map)
            for ind, _ in enumerate(min_zero_map[stop_ind::]): # iterating through each decap in min map
                holder = copy.deepcopy(min_zero_map) # make copy
                if min_zero_map[ind + stop_ind] != 0: # if port is not empty
                    holder[ind + stop_ind] = 0  # make port empty
                    obj = self.f(holder)
                    if obj < self.best_function:
                        # if # of capacitors decrease and target met, overwrite min zero map
                        min_zero_map = copy.deepcopy(holder) # update to better map
                        # improve_bool still true
                        stop_ind = ind
                        break
                    else:
                        holder = copy.deepcopy(min_zero_map)
            new_min = len(min_zero_map) - np.count_nonzero(min_zero_map) # used to set improve bool
            if new_min > current_min:
                print('After check, number of capacitors decreased. Min decap layout is', min_zero_map)
                print('Checking Again....')
                improve_bool = True # not needed but helps me with clarity
            else:
                print('After check, number of capacitors did not decrease.')
                improve_bool = False # score did not improve, set improve_bool to false. break out of loop
        return min_zero_map

    def sweep_check2(self, min_zero_map):  # should use this at some point
        improve_bool = True
        stop_ind = 0
        remove_order = np.flip(np.argsort(min_zero_map))
        while improve_bool is True:
            current_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # number of caps in the initial min_zero_map
            print('Before check, decap map is:', min_zero_map)
            for ind, _ in enumerate(remove_order[stop_ind::]):  # iterating through each decap in min map
                holder = copy.deepcopy(min_zero_map)  # make copy
                if min_zero_map[remove_order[ind + stop_ind]] != 0:  # if port is not empty
                    holder[remove_order[ind + stop_ind]] = 0  # make port empty
                    obj = self.f(holder)
                    if obj < self.best_function:
                        # if # of capacitors decrease and target met, overwrite min zero map
                        min_zero_map = copy.deepcopy(holder)  # update to better map
                        # improve_bool still true
                        stop_ind = ind
                        break
                    else:
                        holder = copy.deepcopy(min_zero_map)
            new_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # used to set improve bool
            if new_min > current_min:
                print('After check, number of capacitors decreased. Min decap layout is', min_zero_map)
                print('Checking Again....')
                improve_bool = True  # not needed but helps me with clarity
            else:
                print('After check, number of capacitors did not decrease.')
                improve_bool = False  # score did not improve, set improve_bool to false. break out of loop
        return min_zero_map

    def sweep_check3(self, min_zero_map):

        if self.L_mat is None:
            raise ValueError('No L Matrix Provided')
        improve_bool = True
        stop_ind = 0

        print('Calculate Port Removal Priority')
        ports_shorted = np.nonzero(min_zero_map)[0]
        remove_order = self.reduce_via_short(self.L_mat, ports_shorted)
        holder = np.in1d(remove_order, self.ports_to_keep)
        remove_order = remove_order[np.where(holder == 0)]
        print('Port Removal Order:', remove_order)

        while improve_bool is True:
            current_min = len(min_zero_map) - np.count_nonzero(
                min_zero_map)  # number of caps in the initial min_zero_map
            print('Before check, decap map is:', min_zero_map)
            for ind, _ in enumerate(remove_order[stop_ind::]):  # iterating through each decap in min map
                holder = copy.deepcopy(min_zero_map)  # make copy
                if min_zero_map[remove_order[ind + stop_ind]] != 0:  # if port is not empty
                    holder[remove_order[ind + stop_ind]] = 0  # make port empty
                    obj = self.f(holder)
                    if obj < self.best_function:
                        # if # of capacitors decrease and target met, overwrite min zero map
                        min_zero_map = copy.deepcopy(holder)  # update to better map
                        # improve_bool still true
                        stop_ind = ind
                        break
                    else:
                        holder = copy.deepcopy(min_zero_map)
            new_min = len(min_zero_map) - np.count_nonzero(min_zero_map)  # used to set improve bool
            if new_min > current_min:
                print('After check, number of capacitors decreased. Min decap layout is', min_zero_map)
                print('Checking Again....')
                improve_bool = True  # not needed but helps me with clarity
            else:
                print('After check, number of capacitors did not decrease.')
                improve_bool = False  # score did not improve, set improve_bool to false. break out of loop
        return min_zero_map

    def sweep_check4(self, sol, short_prio):

        improve_bool = True
        stop_ind = 0
        remove_order = np.copy(short_prio)

        while improve_bool is True:
            current_min = len(sol) - np.count_nonzero(sol)  # number of caps in the initial min_zero_map
            print('Before check, decap map is:', sol)
            for ind, _ in enumerate(remove_order[stop_ind::]):  # iterating through each decap in min map
                holder = copy.deepcopy(sol)  # make copy
                if sol[remove_order[ind + stop_ind]] != 0:  # if port is not empty
                    holder[remove_order[ind + stop_ind]] = 0  # make port empty
                    obj = self.f(holder)
                    if obj < self.best_function:
                        # if # of capacitors decrease and target met, overwrite min zero map
                        sol = copy.deepcopy(holder)  # update to better map
                        # improve_bool still true
                        stop_ind = ind
                        break
                    else:
                        holder = copy.deepcopy(sol)
            new_min = len(sol) - np.count_nonzero(sol)  # used to set improve bool
            if new_min > current_min:
                print('After check, number of capacitors decreased. Min decap layout is', sol)
                print('Checking Again....')
                improve_bool = True  # not needed but helps me with clarity
            else:
                print('After check, number of capacitors did not decrease.')
                improve_bool = False  # score did not improve, set improve_bool to false. break out of loop
        return sol

    def percentage_pop(self, per, num_caps, pop, size_var):
        # initialize population
        new_pop = np.copy(pop)

        # randomly generate population with uniform distribution
        for p in range(0, self.pop_s):
            locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
            for i in locs:
                chance = random.random()
                if chance <= per:
                    new_pop[p][i] = random.randint(1, 3)
                else:
                    new_pop[p][i] = random.randint(4, 10)
        return new_pop

    def percentage_pop2(self, per, num_caps, pop, size_var, short_prio = None):
        # initialize population
        new_pop = np.copy(pop)
        fill_prio = None
        if short_prio is not None:
            fill_prio = np.copy(short_prio)
        for p in range(0, round(self.pop_s)):
            if fill_prio is not None:
                locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
            else:
                locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
            ports_fill = random.sample(list(locs), round(per * len(locs)))
            for i in locs:
                if i in ports_fill:
                    new_pop[p][i] = random.randint(1,3)
                else:
                    new_pop[p][i] = random.randint(4,10)
        return new_pop

    def percentage_pop3(self, per, num_caps, pop, size_var, short_prio = None):
        # initialize population
        new_pop = np.copy(pop)
        fill_prio = None
        if short_prio is not None:
            fill_prio = np.copy(short_prio)

        for p in range(0, round(self.pop_s)):
            if fill_prio is not None:
                locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
            else:
                locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
            ports_fill_g1 = locs[0: round(per * len(locs))]
            for i in locs:
                if i in ports_fill_g1:
                    new_pop[p][i] = random.randint(1,3)
                else:
                    new_pop[p][i] = random.randint(4,10)
        return new_pop


    def percentage_pop_small_caps(self, per, num_caps, pop, size_var, short_prio = None):

        #Places small capacitance caps in best ports
        # initialize population
        new_pop = np.copy(pop)
        fill_prio = None
        if short_prio is not None:
            fill_prio = np.copy(short_prio)

        for p in range(0, round(self.pop_s)):
            if fill_prio is not None:
                locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
            else:
                locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
            ports_fill_g1 = locs[0: round(per * len(locs))]
            for i in locs:
                if i in ports_fill_g1:
                    new_pop[p][i] = random.randint(1,3)
                else:
                    new_pop[p][i] = random.randint(4,10)
        return new_pop

    def percentage_pop_ran_caps(self, per, num_caps, pop, size_var, short_prio=None):

        # Randomly distrbutes caps over the port priority
        # initialize population
        new_pop = np.copy(pop)
        fill_prio = None
        if short_prio is not None:
            fill_prio = np.copy(short_prio)

        for p in range(0, round(self.pop_s)):
            if fill_prio is not None:
                locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
            else:
                locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
            ports_fill = random.sample(list(locs), round(per * len(locs)))
            for i in locs:
                if i in ports_fill:
                    new_pop[p][i] = random.randint(1, 3)
                else:
                    new_pop[p][i] = random.randint(4, 10)
        return new_pop

    def percentage_pop_small_caps_test(self, per, num_caps, pop, size_var, short_prio=None):

        # Places small capacitance caps in best ports
        # initialize population
        new_pop = np.copy(pop)
        fill_prio = None
        if short_prio is not None:
            fill_prio = np.copy(short_prio)

        for p in range(0, round(self.pop_s)):
            if fill_prio is not None:
                locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
            else:
                locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
            ports_fill_g1 = locs[0: round(per * len(locs))]
            new_pop[p][np.s_[ports_fill_g1[0:11]]] = 1
            num_left = len(ports_fill_g1) - 11
            for i in range(num_left):
                new_pop[p][ports_fill_g1[11 + i]] = random.randint(1,3)
            other_caps_ports = locs[len(ports_fill_g1):-1]
            for i in other_caps_ports:
                new_pop[p][i] = random.randint(4,10)
        return new_pop

    def roulette_pop(self, per, num_caps, pop, size_var, short_prio=None):

        # Places small capacitance caps in best ports
        # initialize population
        new_pop = np.copy(pop)
        fill_prio = None
        if short_prio is not None:
            fill_prio = np.copy(short_prio)

        group_1 = np.array([39,22,17,16,8,10,9,10,1,1])
        group_1_sum = np.cumsum(group_1)
        print(group_1_sum)
        group_2 = np.array([1,36,38,46,31,19,20,17,8,12])
        group_2_sum = np.cumsum(group_2)
        cap_array = list(range(1,11))
        for p in range(0, round(self.pop_s)):
            if fill_prio is not None:
                locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
            else:
                locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
            ports_fill_g1 = locs[0: round(per * len(locs))]

            for i in locs:
                if i in ports_fill_g1:
                    roulette_value = random.randint(0, group_1_sum[-1])
                    cap = cap_array[np.searchsorted(group_1_sum, roulette_value)]
                    new_pop[p][i] = cap
                else:
                    roulette_value = random.randint(0, group_2_sum[-1])
                    cap = cap_array[np.searchsorted(group_2_sum, roulette_value)]
                    new_pop[p][i] = cap
        return new_pop

    def R_target_group(self, num_caps, pop, size_var, caps_group_1, num_group_1, short_prio=None):

        # Places small capacitance caps in best ports
        # initialize population
        new_pop = np.copy(pop)
        fill_prio = None
        if short_prio is not None:
            fill_prio = np.copy(short_prio)  # reverses priority to short to get ports to not short


        if np.shape(caps_group_1)[0] == 1:  # Edge case where there is only 1 decap that can meet the last freq pt
            for p in range(0, round(self.pop_s)):
                if fill_prio is not None:
                    locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
                else:
                    locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
                for i in range(num_group_1[0]):
                    new_pop[p][locs[i]] = caps_group_1[0]
                for i in locs[num_group_1[0]::]:
                    new_pop[p][i] = np.random.randint(np.min(self.var_bound[0]) + 1, np.max(self.var_bound[0]) + 1)

        else:
            num_group_1_copy = np.copy(num_group_1)
            cum_sum_roulette = np.abs(num_group_1_copy - np.max(num_group_1_copy))
            for i in range(np.shape(cum_sum_roulette)[0]):
                if cum_sum_roulette[i] == 0:
                    cum_sum_roulette[i] = 1

            cum_sum_roulette = np.cumsum(cum_sum_roulette)
            group_1_rep = np.cumsum(num_group_1_copy)[-1] + self.dim * (np.max(self.var_bound[0]) - np.shape(num_group_1_copy)[0])
            group_1_rep = group_1_rep/  (
                                            self.dim * (np.max(self.var_bound[0]))
                                        )
            for p in range(0, round(self.pop_s)):
                if fill_prio is not None:
                    locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
                else:
                    locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
                ports_fill_g1 = locs[0: round(group_1_rep * len(locs))]

                for i in locs:
                    if i in ports_fill_g1:
                        roulette_value = random.randint(0, cum_sum_roulette[-1])
                        cap = caps_group_1[np.searchsorted(cum_sum_roulette, roulette_value)]
                        new_pop[p][i] = cap
                    else:
                        new_pop[p][i] = np.random.randint(np.min(self.var_bound[0]) + 1, np.max(self.var_bound[0]) + 1)

        return new_pop

    def RL_target_group(self, num_caps, pop, size_var, caps_group_1, num_group_1, short_prio=None):

        # Places small capacitance caps in best ports
        # initialize population
        new_pop = np.copy(pop)
        fill_prio = None
        if short_prio is not None:
            fill_prio = np.flip(np.copy(short_prio)) # reverses priority to short to get ports to not short


        #### Edge Cases #####
        if np.shape(caps_group_1)[0] == 1:  # Edge case where there is only 1 decap that can meet the last freq pt
            for p in range(0, round(self.pop_s)):
                if fill_prio is not None:
                    locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
                else:
                    locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))
                for i in range(num_group_1[0]):
                    new_pop[p][locs[i]] = caps_group_1[0]
                for i in locs[num_group_1[0]::]:
                    new_pop[p][i] = np.random.randint(np.min(self.var_bound[0]) + 1, np.max(self.var_bound[0]) + 1)


        #### Standard Cases ####
        else:
            num_group_1_copy = np.copy(num_group_1)

            cum_sum_roulette1 = np.copy(num_group_1_copy[0,:])
            cum_sum_roulette2 = np.copy(num_group_1_copy[1,:])


            cum_sum_roulette1 = np.abs(cum_sum_roulette1 - np.max(cum_sum_roulette1))
            cum_sum_roulette2 = np.abs(cum_sum_roulette2 - np.max(cum_sum_roulette2))

            for i in range(np.shape(num_group_1_copy)[1]):
                cum_sum_roulette1[i] = 1 if cum_sum_roulette1[i] == 0 else cum_sum_roulette1[i]
                cum_sum_roulette2[i] = 1 if cum_sum_roulette2[i] == 0 else cum_sum_roulette2[i]

            print(cum_sum_roulette1)
            print(cum_sum_roulette2)
            cum_sum_roulette1 = np.cumsum(cum_sum_roulette1)
            cum_sum_roulette2 = np.cumsum(cum_sum_roulette2)
            print(cum_sum_roulette1)
            print(cum_sum_roulette2)


            group_1_rep = np.sum(num_group_1_copy[0]) / (np.sum(num_group_1_copy[0]) + np.sum(num_group_1_copy[1]))

            # print(cum_sum_roulette1)
            # print(cum_sum_roulette2)
            # print(group_1_rep)

            # cum_sum_roulette = np.cumsum(cum_sum_roulette1 + cum_sum_roulette2)
            # group_1_rep = np.sum(num_group_1_copy[0]) + np.sum(num_group_1_copy[1]) \
            #               + self.dim * (np.max(self.var_bound[0]) - np.shape(caps_group_1)[0])
            # group_1_rep = group_1_rep / (
            #         2 * self.dim * (np.max(self.var_bound[0]))
            # )


            for p in range(0, round(self.pop_s)):
                if fill_prio is not None:
                    locs = fill_prio[0:random.randint(num_caps, num_caps + size_var)]
                else:
                    locs = random.sample(range(self.dim), random.randint(num_caps, num_caps + size_var))

                caps_to_fill_g1 = np.zeros((round(group_1_rep * np.shape(locs)[0])), dtype = int)
                caps_to_fill_g2 = np.zeros((np.shape(locs)[0] - round(group_1_rep * np.shape(locs)[0])), dtype=int)


                for i in range(np.shape(caps_to_fill_g1)[0]):
                    roulette_value = random.randint(0, cum_sum_roulette1[-1])
                    caps_to_fill_g1[i] = caps_group_1[0, np.searchsorted(cum_sum_roulette1, roulette_value)]
                for i in range(np.shape(caps_to_fill_g2)[0]):
                    roulette_value = random.randint(0, cum_sum_roulette2[-1])
                    caps_to_fill_g2[i] = caps_group_1[1, np.searchsorted(cum_sum_roulette2, roulette_value)]


                caps_to_fill = np.concatenate((caps_to_fill_g1, caps_to_fill_g2))
                #caps_to_fill = np.sort(caps_to_fill)
                new_pop[p][locs] = caps_to_fill

        return new_pop

    def pop_based_off_sol(self, model_sol, pop, condtions = None, cap_mod = None):

        # this function will make a new solution based off a particular solution

        model_copy = model_sol.copy()
        new_pop = pop.copy()

        sols_shifted = round(np.shape(pop)[0]/2)
        #sols_mutted = np.shape(pop)[0] - np.round(np.shape(pop)[0]/2)

        new_pop[:,0:-1] = model_copy.copy()

        for i in range(sols_shifted):
            new_pop[i,0:-1] = self.shift_mut(self.short_prio, new_pop[i,0:-1], .1, move_max = 10)

        for i in range(sols_shifted, np.shape(pop)[0]):
            new_pop[i,0:-1] = self.cust_mut(new_pop[i,0:-1], .1, mut_zeros=False)

        return new_pop




    ##### Temporary Save
# def pop_size_control(self, pop, size, random_change=False, force_empty=None, num_cap_mod=0):
#     # this function will take a population, pop, and changes the number of capacitors for solutions in the population
#     # to a specific number
#
#     # ACTUALLY POP IS JUST ONE SOLUtioN. 1 DECAP MAP
#
#     # for the PDN capacitor selection, a 0 represents an empty port.
#     # we want to control max # of capacitors (non-zero) can be in the solution
#     # Size is the max # of capacitors allowed in the solution
#
#     # later may change so size is a range but not rn
#     # Also to note, from default code, pop has dimension [0:self.dim+1]
#     # 0 to self.dim contains the genes
#     # the self.dim + 1 should be the evaluation/score
#
#     # pop is the population object
#     # size is an integer from 1 to self.dim
#
#     # There might be issues with num_cap_mod causing you to have < 0 caps. I think anyways.
#     # not an issue if num_cap_mod = 0
#     if not isinstance(size, int):
#         raise TypeError('The "Size" of the Solution Must be an Integer In [1, #of Genes]')
#
#     if size < 1 or size > self.dim:
#         raise ValueError('Size of a Solution Must Be Between 1 or the # of Genes (# of Ports')
#
#     new_pop = pop.copy()
#     # I think i can get away with just copy here, as long as I pass only 1 solution (which it currently is and is hould
#
#     if np.ndim(new_pop) == 1:  # if you are passing only one solution and not the entire population
#         new_pop = np.array([new_pop])
#
#     for i in range(np.shape(new_pop)[0]):
#         new_chrome = new_pop[i][0:self.dim].copy()  # copies the genes
#
#         if force_empty is not None:  # if there are ports we need to force empty
#             for j in force_empty:
#                 new_chrome[j] = 0
#
#         current_size = np.count_nonzero(new_chrome)  # number of caps in current solution
#         caps_locations = np.nonzero(new_chrome)[0]  # gets locations where ports are not empty
#         caps = new_chrome[caps_locations]  # gets capacitors at non-empty ports
#
#         if current_size > size:
#             num_to_empty = current_size - size + num_cap_mod
#             if current_size - num_to_empty < size - self.size_variation:
#                 num_to_empty = current_size - size
#
#             ports_to_empty = np.random.choice(caps_locations, size=num_to_empty, replace=False)
#             for j in ports_to_empty:
#                 new_chrome[j] = 0
#             new_pop[i][0:self.dim] = new_chrome.copy()
#
#         elif current_size < size - self.size_variation:
#             empty_locations = np.nonzero(new_chrome == 0)[0]  # get port locations that have no decaps
#             if force_empty is not None:  # if there are ports we need to keep empty
#                 force_empty_rev = np.flip(force_empty)  # get indices of ports that must be forced empty
#                 for j in force_empty_rev:
#                     empty_locations = np.delete(empty_locations,
#                                                 np.where(empty_locations == j)[0])  # remove them from the fill list
#
#             num_to_add = size - current_size + num_cap_mod
#             if current_size + num_to_add >= size:
#                 num_to_add = size - current_size
#
#             ports_to_fill = np.random.choice(empty_locations, size=num_to_add, replace=False)
#             if random_change:
#                 for j in ports_to_fill:
#                     new_cap = np.random.randint(np.min(self.var_bound[0]) + 1, np.max(self.var_bound[0]) + 1)
#                     new_chrome[j] = new_cap
#                 # new_pop[i][0:self.dim] = new_chrome.copy()
#             else:
#                 # replace with a cap already in solution rather than randomly out of the entire range
#                 for j in ports_to_fill:
#                     new_cap = caps[np.random.randint(0, current_size)]
#                     new_chrome[j] = new_cap
#
#                 new_pop[i][0:self.dim] = new_chrome.copy()
#
#     if np.shape(new_pop)[0] == 1:
#         new_pop = new_pop[0]
#
#     return new_pop
