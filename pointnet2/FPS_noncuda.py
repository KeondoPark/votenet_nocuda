import torch
import pointnet2_utils
import pytorch_utils as pt_utils

import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, os.pardir)
print(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pc_util import random_sampling, read_ply

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)    
    point_cloud = random_sampling(point_cloud, 20000)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

demo_dir = os.path.join(BASE_DIR, 'demo_files/sunrgbd_results')
pc_path = os.path.join(demo_dir, '000000_pc.ply')

point_cloud = read_ply(pc_path)
pc = preprocess_point_cloud(point_cloud)
print('Loaded point cloud data: %s'%(pc_path))

xyz = {'point_clouds': torch.from_numpy(pc).to(device)}
pointnet2_utils.furthest_point_sample(xyz, 20000)
