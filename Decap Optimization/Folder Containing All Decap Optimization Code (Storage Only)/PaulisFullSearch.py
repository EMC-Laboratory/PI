import copy
import numpy as np
from config2 import Config
import ShapePDN as pdn
import matplotlib.pyplot as plt
import time
def cost_function(solution_z, opt):

    score = 0

    abs_map_z = copy.deepcopy(solution_z)
    abs_map_z = np.abs(abs_map_z)
    score_holder = 0
    add_score = 0

    for index in range(len(abs_map_z)):
        add_score = abs(abs_map_z[index] - opt.ztarget[index]) if abs(abs_map_z[index]) >= abs(opt.ztarget[index]) else 0
        score_holder = score_holder + add_score
    pts_above = np.count_nonzero(np.greater(abs_map_z,opt.ztarget))

    pts_above = pts_above if pts_above > 0 else 1
    score = score_holder/pts_above

    return score

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

def OptionsInit():
    # Get settings
    opt = Config()
    return opt

def decap_objects(opt):
    # Store capacitor library as their impedances
    cap_objs = [pdn.select_decap(i, opt) for i in range(1,opt.num_decaps+1)] # list of shorted capacitors, 1 to 10, high end is not inclusive [x,y) so need + 1
    cap_objs_z = copy.deepcopy(cap_objs)
    cap_objs_z = [cap_objs_z[i].z for i in range(opt.num_decaps)]
    return cap_objs, cap_objs_z

opt = OptionsInit()
cap_objs, cap_objs_z = decap_objects(opt)

# Impedance Settings
R = .015
Zmax = .015
freq, z_target = get_target_z_RL(R, Zmax, opt, fstart=1e4, fstop=20e6, nf=201 interp='log')
opt.ztarget = copy.deepcopy(z_target)



ports_to_fill = [i for i in range(0,14)]
best_score = 1e9
sol_found = False

solution = [0] * opt.decap_ports # holder for to keep track of each capacitor added
final_sol = [] # holder for the solution that satisfies the target if it exists
current_best_sol = [] # holder for the current best solution

input_z = opt.input_net.z

f = open("Paulis Full Search Test.txt", 'a')


while len(ports_to_fill) != 0 and sol_found == False:

    for i in ports_to_fill:

        for j in range(opt.num_decaps):
            temp = solution.copy()
            temp[i] = j+1
            print(temp)
            temp_z = pdn.new_connect_n_decap(input_z, temp, cap_objs_z, opt )
            score = cost_function(temp_z, opt)

            if score == 0:
                best_score = score
                final_sol = temp.copy()
                current_best_sol = temp.copy()
                sol_found = True
                break

            elif score < best_score:
                best_score = score
                current_best_sol = temp.copy()
        if best_score == 0:
            break
    index_to_remove = [x for x,y in enumerate(current_best_sol) if x in ports_to_fill and y != 0][0]
    ports_to_fill.remove(index_to_remove)
    print(index_to_remove, 'here')
    print(ports_to_fill)
    #time.sleep(5)
    solution = current_best_sol.copy()

    final_z = pdn.new_connect_n_decap(input_z, solution, cap_objs_z, opt)
    plt.loglog(opt.freq, z_target, '--', color='black')
    plt.loglog(opt.freq, np.abs(final_z))
    plt.title('Best Solution', fontsize=16)
    plt.xlabel('Freq in Hz', fontsize=16)
    plt.ylabel('Impedance in Ohms', fontsize=16)
    plt.legend(['Target Z', 'Best Solution {}'.format(solution)], prop={'size': 14})
    plt.grid(True, which='Both')
    plt.show()

    best_score = 1e9

if sol_found == True:
    print('Solution found satisfying target is', final_sol)
    final_z = pdn.new_connect_n_decap(input_z, final_sol, cap_objs_z, opt)

    f.write('Solution Found. Num caps =' + str(np.count_nonzero(final_sol)) + "\n")
    f.write(str(final_sol))
    f.write("\n,\n")

    plt.loglog(freq,np.abs(final_z))
    plt.loglog(freq, opt.ztarget)
    plt.show()

else:
    print('No solution found satisfying target. All capacitor ports are filled. Solution at end is:', solution)
    final_z = pdn.new_connect_n_decap(input_z, solution, cap_objs_z, opt)

    f.write('No Solution Found. Num caps =' + str(np.count_nonzero(solution)) + "\n")
    f.write(str(solution))
    f.write("\n,\n")

    plt.loglog(freq, np.abs(final_z))
    plt.loglog(freq, opt.ztarget)
    plt.show()

f.close()