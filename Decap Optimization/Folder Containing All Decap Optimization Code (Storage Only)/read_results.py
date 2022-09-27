import ast

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
def get_results(f):
    lines = f.readlines()
    time_taken = 0
    final_num_caps = 0
    target_met = 0
    final_sol = 0
    best_num_caps = 0
    best_sol = 0
    conv = 0

    for line_num, i in enumerate(lines):
        if line_num == 0:
            x = i.split('=')
            time_taken = x[-1].rstrip()
            time_taken = ast.literal_eval(time_taken)
        if line_num == 1:
            x = i.split('=')
            final_num_caps = x[-1].rstrip()
            final_num_caps = ast.literal_eval(final_num_caps)
        if line_num == 2:
            x = i.split(' ')
            target_met = 1 if x[-1].rstrip() == 'True' else 0
        if line_num == 3:
            final_sol = i.rstrip()
            final_sol = np.asarray(ast.literal_eval(final_sol))
        if line_num == 4:
            x = i.split('=')
            best_num_caps = x[-1].rstrip()
            best_num_caps = ast.literal_eval(best_num_caps)
        if line_num == 5:
            best_sol = i.rstrip()
            best_sol = np.asarray(ast.literal_eval(best_sol))
        if line_num == 7:
            conv = i.rstrip()
            conv = np.asarray(ast.literal_eval(conv))
    return time_taken, final_num_caps, target_met, final_sol, best_num_caps, best_sol, conv

results_folder = 'Compare Different Methods Results'

method_types = ['Old', 'New']
num_ports = [25,50,75]
num_targets = [1,2,3]
num_runs = [1,2,3,4,5]
num_boards = [1,2,3]

#time_taken, final_num_caps, target_met, final_sol, best_num_caps, best_sol, conv

ostore_time = []
ostore_fcaps = []
ostore_tar_met = []
ostore_fsol = []
ostore_bcaps = []
ostore_bsol = []
ostore_conv = []

fstore_time = []
fstore_fcaps = []
fstore_tar_met = []
fstore_fsol = []
fstore_bcaps = []
fstore_bsol = []
fstore_conv = []


for i in method_types:
    for j in num_ports:
        for k in num_boards:
            for l in num_targets:
                for m in num_runs:
                    sub_path = 'Compare Methods {} GA Conv {} Ports Board {}'.format(i, j, k)
                    target_txt = 'Target Impedance {} Conv {}.txt'.format(l,m)
                    search_path = os.path.join(results_folder, sub_path, target_txt)
                    f = open(search_path,'r')
                    [time_taken, final_num_caps, target_met, final_sol, best_num_caps, best_sol, conv] =\
                    get_results(f)
                    f.close()
                    if i == 'Old':
                        ostore_time.append(time_taken)
                        ostore_fcaps.append(final_num_caps)
                        ostore_tar_met.append(target_met)
                        ostore_fsol.append(final_sol)
                        ostore_bcaps.append(best_num_caps)
                        ostore_bsol.append(best_sol)
                        ostore_conv.append(conv)
                    else:
                        fstore_time.append(time_taken)
                        fstore_fcaps.append(final_num_caps)
                        fstore_tar_met.append(target_met)
                        fstore_fsol.append(final_sol)
                        fstore_bcaps.append(best_num_caps)
                        fstore_bsol.append(best_sol)
                        fstore_conv.append(conv)
                    print(i, j,k,l,m, 'done')
gen_initial_old = np.zeros((1,len(num_ports) * len(num_targets) * len(num_boards) * len(num_runs)))
gen_initial_new = np.zeros((1,len(num_ports) * len(num_targets) * len(num_boards) * len(num_runs)))


# for ind,i in enumerate(ostore_conv):
#     gen_initial = np.where(i < 0)[0][0]
#     print('here',gen_initial)
#     gen_initial_old[0,ind] = gen_initial[0]+1 if sum(gen_initial) != 0 else 100

# for ind, i in enumerate(fstore_conv):
#     gen_initial = np.where(i < 0)[0]
#     print('here',gen_initial)
#     gen_initial_new[0,ind] = gen_initial[0]+1 if sum(gen_initial) != 0 else 100


plt.scatter(range(135), ostore_bcaps, linewidths = 3)
plt.scatter(range(135), fstore_fcaps,  marker = 'x', c = 'r', linewidths=3)
plt.xlabel('Impedance Target/Algorithm Run', fontsize = 22)
plt.ylabel('Number of Capacitors in Best Solution Found', fontsize = 22)
plt.title('Comparison Results of GA_ref vs Augmented GA', fontsize = 22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.axvline(x = 3*3*5,c = 'black',ls ='--')
plt.axvline(x = 3*3*5*2,c='black',ls ='--')
plt.axvline(x = 3*3*5*3,c='black', ls ='--')
plt.legend(['GA_ref','Augmented GA'], fontsize = 20)
plt.show()