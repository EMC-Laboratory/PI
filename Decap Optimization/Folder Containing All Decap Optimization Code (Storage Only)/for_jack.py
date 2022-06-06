# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:40:37 2020

@author: lingzhang0319
"""

from pdn_class import PDN
import numpy as np
from math import sqrt, pi, sin, cos, log, atan, pow


# This is the first example with an irregularly-shaped power plane
'''brd2 = PDN()
brd2.outer_bd_node = np.array([[0,0],[5,0],[5,5],[10,5],[10,0],[20,0],[20,5],[30,5],[30,0],[40,0],
                              [40,8],[25,20],[25,30],[10,30],[10,15],[0,10],[0,0]])*1e-3
brd2.seg_len = 2e-3
brd2.er = 4.3
brd2.seg_bd()
brd2.ic_via_xy = np.array([[10,10],[12,10]])*1e-3
brd2.ic_via_type = np.array([1,0])
brd2.stackup = np.array([0,0,1,0,1])
brd2.die_t = np.array([0.3e-3,0.3e-3,0.5e-3,0.5e-3])
brd2.via_r = 0.1e-3
brd2.calc_mat_wo_decap()
brd2.add_decap(26e-3,8e-3,28e-3,8e-3,9)
brd2.add_decap(27e-3,15e-3,27e-3,13e-3,7)
brd2.add_decap(13e-3,25e-3,13e-3,27e-3,5)
brd2.add_decap(21e-3,25e-3,21e-3,27e-3,3)
brd2.add_decap(18e-3,21e-3,18e-3,23e-3,5)'''

# This is another board example with rectangular shape

# defines physical properties (size, pervmitivity, etc) and ic via locations
brd = PDN()
brd.outer_bd_node = np.array([[200,200],[2200,200],[2200,1600],[200,1600],[200,200]])*0.0254*1e-3
brd.seg_len = 5e-3
brd.er = 4.3
brd.seg_bd()
brd.ic_via_xy = np.array([[1100,898],[1140,898],[1180,898],[1100,938],[1140,938],[1180,938],[1100,978],
                           [1140,978],[1180,978],[1100,858],[1140,858],[1180,858],[1060,898],[1060,938],
                           [1060,978],[1100,1018],[1140,1018],[1180,1018],[1220,978],[1220,938],
                           [1220,898],[350,298],[2040,298],[2040,1578],[350,1578]])*0.0254e-3
                            # .0254 * 10^-3 to get into mils? or something
brd.ic_via_type = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
brd.stackup = np.array([0,0,1,0,1])
brd.die_t = np.array([0.3e-3,0.3e-3,0.5e-3,0.5e-3])
brd.via_r = 0.1e-3
brd.calc_mat_wo_decap()

# From what I can tell, the funciton brd.add_decap will:
    # Update the z parameters of the board with the added via slot for the capacitor (w/o actually inserting a cap in)
        # saves to brd.zmergeIC_non_decap
    # And update the z parameters for the board with the added via slot and capacitor put in.
        # brd.zmergeIC_with_decap saved to there
    # So i can just use the no decap files the way I've been doing before and it should work fine.
    # You change the number of decap ports/sNp files # of ports by adding and changing # of decap ports
brd.add_decap(800*0.0254e-3,800*0.0254e-3,800*0.0254e-3+2e-3,800*0.0254e-3,9)
brd.add_decap(1600*0.0254e-3,1200*0.0254e-3,1600*0.0254e-3+2e-3,1200*0.0254e-3,8)
brd.add_decap(800*0.0254e-3,1200*0.0254e-3,800*0.0254e-3+2e-3,1200*0.0254e-3,7)
brd.add_decap(1200*0.0254e-3,1200*0.0254e-3,1200*0.0254e-3+2e-3,1200*0.0254e-3,6)
brd.add_decap(1400*0.0254e-3,800*0.0254e-3,1400*0.0254e-3+2e-3,800*0.0254e-3,5)
brd.add_decap(1600*0.0254e-3,600*0.0254e-3,1600*0.0254e-3+2e-3,600*0.0254e-3,1)
brd.add_decap(1800*0.0254e-3,400*0.0254e-3,1800*0.0254e-3+2e-3,400*0.0254e-3,9)
brd.add_decap(1800*0.0254e-3,600*0.0254e-3,1800*0.0254e-3+2e-3,600*0.0254e-3,9)
brd.add_decap(1200*0.0254e-3,600*0.0254e-3,1200*0.0254e-3+2e-3,600*0.0254e-3,9)
brd.add_decap(600*0.0254e-3,1000*0.0254e-3,600*0.0254e-3+2e-3,1000*0.0254e-3,9)
# brd.add_decap(1600*0.0254e-3,1000*0.0254e-3,1600*0.0254e-3+2e-3,1000*0.0254e-3,9)
# brd.add_decap(1400*0.0254e-3,1000*0.0254e-3,1400*0.0254e-3+2e-3,1000*0.0254e-3,9)
# brd.add_decap(1000*0.0254e-3,600*0.0254e-3,1000*0.0254e-3+2e-3,600*0.0254e-3,9)
# brd.add_decap(1000*0.0254e-3,1400*0.0254e-3,1000*0.0254e-3+2e-3,1400*0.0254e-3,9)
# brd.add_decap(600*0.0254e-3,1400*0.0254e-3,600*0.0254e-3+2e-3,1400*0.0254e-3,9)

brd.plot_z(brd.z_mergeIC_no_decap)
brd.plot_z(brd.z_mergeIC_with_decap)

#print((brd.z_mergeIC_no_decap))
#brd.save2s(brd.z_mergeIC_no_decap,'forjack_example')





#### By default
# -*- coding: utf-8 -*-
# """
# Created on Sat Sep 19 15:40:37 2020
#
# @author: lingzhang0319
# """
#
# from pdn_class import PDN
# import numpy as np
# from math import sqrt, pi, sin, cos, log, atan
#
# # This is the first example with an irregularly-shaped power plane
# '''brd2 = PDN()
# brd2.outer_bd_node = np.array([[0,0],[5,0],[5,5],[10,5],[10,0],[20,0],[20,5],[30,5],[30,0],[40,0],
#                               [40,8],[25,20],[25,30],[10,30],[10,15],[0,10],[0,0]])*1e-3
# brd2.seg_len = 2e-3
# brd2.er = 4.3
# brd2.seg_bd()
# brd2.ic_via_xy = np.array([[10,10],[12,10]])*1e-3
# brd2.ic_via_type = np.array([1,0])
# brd2.stackup = np.array([0,0,1,0,1])
# brd2.die_t = np.array([0.3e-3,0.3e-3,0.5e-3,0.5e-3])
# brd2.via_r = 0.1e-3
# brd2.calc_mat_wo_decap()
# brd2.add_decap(26e-3,8e-3,28e-3,8e-3,9)
# brd2.add_decap(27e-3,15e-3,27e-3,13e-3,7)
# brd2.add_decap(13e-3,25e-3,13e-3,27e-3,5)
# brd2.add_decap(21e-3,25e-3,21e-3,27e-3,3)
# brd2.add_decap(18e-3,21e-3,18e-3,23e-3,5)'''
#
# # This is another board example with rectangular shape
#
# # defines physical properties (size, pervmitivity, etc) and ic via locations
# brd = PDN()
# brd.outer_bd_node = np.array([[200,200],[2200,200],[2200,1600],[200,1600],[200,200]])*0.0254*1e-3
# brd.seg_len = 5e-3
# brd.er = 4.3
# brd.seg_bd()
# brd.ic_via_xy = np.array([[1100,898],[1140,898],[1180,898],[1100,938],[1140,938],[1180,938],[1100,978],
#                            [1140,978],[1180,978],[1100,858],[1140,858],[1180,858],[1060,898],[1060,938],
#                            [1060,978],[1100,1018],[1140,1018],[1180,1018],[1220,978],[1220,938],
#                            [1220,898],[350,298],[2040,298],[2040,1578],[350,1578]])*0.0254e-3
#                             # .0254 * 10^-3 to get into mils? or something
# brd.ic_via_type = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# brd.stackup = np.array([0,0,1,0,1])
# brd.die_t = np.array([0.3e-3,0.3e-3,0.5e-3,0.5e-3])
# brd.via_r = 0.1e-3
# brd.calc_mat_wo_decap()
#
#
#
# # places a location where you can put a capacitor
#
#
# # From what I can tell, the funciton brd.add_decap will:
#     # Update the z parameters of the board with the added via slot for the capacitor (w/o actually inserting a cap in)
#         # saves to brd.zmergeIC_non_decap
#     # And update the z parameters for the board with the added via slot and capacitor put in.
#         # brd.zmergeIC_with_decap saved to there
#     # So i can just use the no decap files the way I've been doing before and it should work fine.
#     # You change the number of decap ports/sNp files # of ports by adding and changing # of decap ports
# brd.add_decap(800*0.0254e-3,800*0.0254e-3,800*0.0254e-3+2e-3,800*0.0254e-3,9)
# brd.add_decap(1600*0.0254e-3,1200*0.0254e-3,1600*0.0254e-3+2e-3,1200*0.0254e-3,8)
# brd.add_decap(800*0.0254e-3,1200*0.0254e-3,800*0.0254e-3+2e-3,1200*0.0254e-3,7)
# brd.add_decap(1200*0.0254e-3,1200*0.0254e-3,1200*0.0254e-3+2e-3,1200*0.0254e-3,6)
# brd.add_decap(1400*0.0254e-3,800*0.0254e-3,1400*0.0254e-3+2e-3,800*0.0254e-3,5)
# brd.add_decap(1600*0.0254e-3,600*0.0254e-3,1600*0.0254e-3+2e-3,600*0.0254e-3,1)
# brd.add_decap(1800*0.0254e-3,400*0.0254e-3,1800*0.0254e-3+2e-3,400*0.0254e-3,9)
# brd.add_decap(1800*0.0254e-3,600*0.0254e-3,1800*0.0254e-3+2e-3,600*0.0254e-3,9)
# brd.add_decap(1200*0.0254e-3,600*0.0254e-3,1200*0.0254e-3+2e-3,600*0.0254e-3,9)
# brd.add_decap(600*0.0254e-3,1000*0.0254e-3,600*0.0254e-3+2e-3,1000*0.0254e-3,9)
# # brd.add_decap(1600*0.0254e-3,1000*0.0254e-3,1600*0.0254e-3+2e-3,1000*0.0254e-3,9)
# # brd.add_decap(1400*0.0254e-3,1000*0.0254e-3,1400*0.0254e-3+2e-3,1000*0.0254e-3,9)
# # brd.add_decap(1000*0.0254e-3,600*0.0254e-3,1000*0.0254e-3+2e-3,600*0.0254e-3,9)
# # brd.add_decap(1000*0.0254e-3,1400*0.0254e-3,1000*0.0254e-3+2e-3,1400*0.0254e-3,9)
# # brd.add_decap(600*0.0254e-3,1400*0.0254e-3,600*0.0254e-3+2e-3,1400*0.0254e-3,9)
#
# #brd.plot_z(brd.z_mergeIC_with_decap)
# brd.plot_z(brd.z_mergeIC_no_decap)
# #print((brd.z_mergeIC_no_decap))
# #brd.save2s(brd.z_mergeIC_no_decap,'forjack_example')
#

