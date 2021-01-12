import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

import pointnet2_utils
import pytorch_utils as pt_utils

import numpy as np
import time

BASE_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
ROOT_DIR=BASE_DIR
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))
sys.path.append(os.path.join(ROOT_DIR,'models'))

from typing import List
from pc_util import read_ply, random_sampling

class pointnet(nn.Module):

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            bn: bool = True,
            use_xyz: bool = True
            ):
        super().__init__()
        self.npoint = npoint
        self.mlp_module = None
        self.use_xyz = use_xyz

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)        


    def forward(self, xyz):        
        return self.mlp_module(xyz)


def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)  
    
    point_cloud = random_sampling(point_cloud, 20000)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

if __name__ == '__main__':

    demo_dir = os.path.join(BASE_DIR, 'demo_files')
    sys.path.append(os.path.join(ROOT_DIR,'sunrgbd'))
    from sunrgbd_detection_dataset import DC
    #checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
    pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')

    point_cloud = read_ply(pc_path)
    pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))

    tic = time.time() 
    net = pointnet(bn=True, mlp=[0,64,128,256], npoint=2048, use_xyz=True)     
    pc_tensor = torch.from_numpy(pc[...,:3])
    pc_trans = pc_tensor.transpose(1,2).unsqueeze(2)
    out = net(pc_trans)
    toc = time.time()    
    print(out)
    print("Calculation time: %f"%(toc - tic))
