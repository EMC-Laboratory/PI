import scipy.io
import copy
import numpy
import math
import os
import matplotlib.pyplot as plt
# absolutely none of theese are optimized for speed or anything

def save_2_mat(list_of_info, headings_for_info, file_name):


    #Required Package: scipy.io, copy
    info = copy.deepcopy(list_of_info)
    names = copy.deepcopy(headings_for_info)
    mat = dict()
    for i in range(len(list_of_info)):
        mat[names[i]] = info[i]
    scipy.io.savemat(file_name + ".mat", mat)


def save_lmat_from_npz():
    pass


def pts_to_bd(sxy,accuracy= 0.01):

    # Required Package: numpy, math

    # this function takes the points that make a boundary and returns the points where the curve changes direction
    # probably works

    # sxy are the points that make up a boundary. Must be in the order the curve is drawn to have any meaning.

    # this function calculates angles to determine directions. Due to precision errors, angles won't be exactly equal.
    # for 3 points x,y,z,  this function calculates changes in direction by comparing the angle from x to y and y to z
    # accuracy is the max percent difference (angle(yz) - angle(xy)) has to be to be considered a change in direction.
    # don't set too small or too big. Depends on precision of math operations.
    # 


    all_segments = numpy.copy(sxy[:, 2:4])
    all_segments = numpy.insert(all_segments, 0, sxy[0, 0:2], axis=0)
    boundary_pts = numpy.empty((1, 2))
    angle = 0
    new_bd_pt_flag = False

    for i in range(numpy.shape(all_segments)[0] - 1):

        # For angle = -1, it is to catch cases where you move perfectly upwards, like the vertical sides of a square
        if new_bd_pt_flag:
            # This part triggers when the previous point, all_segments[i-1], is a boundary point
            # calculate the angle between the current point and that previous boundary point
            if all_segments[i, 0] != all_segments[i-1, 0]:
                angle = math.atan(
                    (all_segments[i, 1] - all_segments[i-1, 1]) / (all_segments[i, 0] - all_segments[i-1, 0]))
            else:
                angle = -1
            new_bd_pt_flag = False

        if i == 0:
            boundary_pts = numpy.append(boundary_pts, all_segments[i])

            if all_segments[i + 1, 0] != all_segments[i, 0]:
                angle = math.atan(
                    (all_segments[i + 1, 1] - all_segments[i, 1]) / (all_segments[i + 1, 0] - all_segments[i, 0]))
            else:
                angle = -1
        else:
            if all_segments[i + 1, 0] != all_segments[i, 0]:
                current_angle = math.atan(
                    (all_segments[i + 1, 1] - all_segments[i, 1]) / (all_segments[i + 1, 0] - all_segments[i, 0]))
            else:
                current_angle = -1
            print(current_angle)
            print(all_segments[i])
            #print(abs( abs(angle) - abs(current_angle)) / ((abs(angle) + abs(current_angle))/2) * 100)
            print(angle,'\n')

            if angle == 0 and current_angle == 0:
                # special case to catch movement in the x direction only
                pass

            # elif angle == current_angle and angle != 0:
            #     if all_segments[i+1,0] < all_segments[i,0] and all_segments[i+1,1] > all_segments[i,0]:
            #         boundary_pts = numpy.append(boundary_pts, all_segments[i])
            #     elif all_segments[i+1,1] < all_segments[i,1] and all_segments[i+1,0] > all_segments[i,0]:
            #         boundary_pts = numpy.append(boundary_pts, all_segments[i])


            elif (abs(abs(angle) - abs(current_angle)) / ((abs(angle) + abs(current_angle)) / 2) * 100) > accuracy:
                boundary_pts = numpy.append(boundary_pts, all_segments[i])
                new_bd_pt_flag = True

        if i == numpy.shape(all_segments)[0] - 1:
            boundary_pts = numpy.append(boundary_pts, all_segments[-1])

    boundary_pts = boundary_pts[2::]
    # The boundary pts are returned as [x1,y1,x2,y2,x3,y3.....] as a list
    # each pair gives a boundary point. Does not close the boundary
    return boundary_pts


# Read files in
save_path = 'Testing for Moving Vias/'
file_name = '1'
save_file_path = os.path.join(save_path, file_name)
sxy = numpy.load(save_file_path + '.npz')['sxy']
bd = pts_to_bd(sxy,0.1)
plt.plot(sxy[:,2],sxy[:,3])
plt.scatter(bd[0::2], bd[1::2], color = 'black')

# save_path = 'Testing for Moving Vias/'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# file_name = 'Recreated 100 Port Boundary'
# save_file_path = os.path.join(save_path, file_name)
# numpy.savez(save_file_path, bd = bd)
#
plt.show()