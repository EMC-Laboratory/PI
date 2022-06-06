import numpy as np
import os
import matplotlib.pyplot as plt

# BASE_PATH = 'new_data/'
BASE_PATH = 'new_data_test/'

if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

n = 10200
z_orig = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['z']
brd_shape_ic = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['brd_shape_ic']
ic_xy_indx = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['ic_xy_indx']
top_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['top_decap_xy_indx']
bot_decap_xy_indx = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['bot_decap_xy_indx']
stackup = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['stackup']
die_t = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['die_t']
sxy = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['sxy']
ic_via_xy = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['ic_via_xy']
vrm_xy_indx = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['vrm_xy_indx']
vrm_loc = np.load(os.path.join(BASE_PATH, str(n) + '.npz'))['vrm_loc']

# Get PWR via locations and create ground via locations
via_dist = 1e-3

x = np.concatenate( ((top_decap_xy_indx[:,0]+0.5)*200e-3/16 - via_dist/2, (bot_decap_xy_indx[:,0]+0.5)*200e-3/16 - via_dist/2)) # x coord
y = np.concatenate( ((top_decap_xy_indx[:,1]+0.5)*200e-3/16,  (bot_decap_xy_indx[:,1]+0.5)*200e-3/16)) # y coord

# ground vias
gx = np.concatenate( ((top_decap_xy_indx[:,0]+0.5)*200e-3/16 + via_dist/2, (bot_decap_xy_indx[:,0]+0.5)*200e-3/16 + via_dist/2)) # x coord
gy = np.concatenate( ((top_decap_xy_indx[:,1]+0.5)*200e-3/16,  (bot_decap_xy_indx[:,1]+0.5)*200e-3/16)) # y coord




f = open("Via Locations and Pwr Shape.txt", "a")
f.write('REGULAR_VIA_MODEL\n\n')
ic_type = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
print(ic_via_xy)
for i in range(len(ic_via_xy)):
    f.write('Via\n')
    f.write('Name\t VIA{}\n'.format(i + 1))
    f.write('X\t {}\n'.format(round(ic_via_xy[i][0], 5) * 1000))  # 1000 times is to convert to mm from m
    f.write('Y\t {}\n'.format(round(ic_via_xy[i][1], 5) * 1000))
    f.write('PadStackName\t Padstack1\n')
    if ic_type[i] == 1:
        f.write('NetName\t PWR\n')
    else:
        f.write('NetName\t GND\n')
    f.write('StartLayerName\t Top\n')
    f.write('EndLayerName\t Bottom\n')
    f.write('EndVia\n\n')


for i in range(len(x)):
    f.write('Via\n')
    f.write('Name\t VIA{}\n'.format(i+1 + len(ic_type)))
    f.write('X\t {}\n'.format(round(x[i],5) * 1000)) # 1000 times is to convert to mm from m
    f.write('Y\t {}\n'.format(round(y[i],5) * 1000))
    f.write('PadStackName\t Padstack1\n')
    f.write('NetName\t PWR\n')
    f.write('StartLayerName\t Top\n')
    f.write('EndLayerName\t Bottom\n')
    f.write('EndVia\n\n')


for i in range(len(gx)):
    f.write('Via\n')
    f.write('Name\t VIA{}\n'.format(i+1 + len(ic_type) + len(x)))
    f.write('X\t {}\n'.format(round(gx[i],5) * 1000)) # 1000 times is to convert to mm from m
    f.write('Y\t {}\n'.format(round(gy[i],5) * 1000))
    f.write('PadStackName\t Padstack1\n')
    f.write('NetName\t GND\n')
    f.write('StartLayerName\t Top\n')
    f.write('EndLayerName\t Bottom\n')
    f.write('EndVia\n\n')

f.write('END_REGULAR_VIA_MODEL\n\n')

# Get board outline and verticies
seg_len = 2e-3
x_vertices = sxy[:,2]
y_verticies = sxy[:,3]

f.write('POWER_NET_SHAPE\n')

for i in range(len(x_vertices)):
    f.write('Vertex\t {}\t {}\n'.format(round(1000*x_vertices[i]),round(1000*y_verticies[i]))) # 1000* is to convert m to mm
f.write('END_POWER_NET_SHAPE\n\n')

f.close()

