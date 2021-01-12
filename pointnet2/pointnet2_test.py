# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Testing customized ops. '''

import torch
from torch.autograd import gradcheck
from torch.autograd import Variable
import numpy as np

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pointnet2_utils

def test_interpolation_grad():
    batch_size = 1
    feat_dim = 2
    m = 4
    feats = torch.randn(batch_size, feat_dim, m, requires_grad=True).float().cuda()
    
    def interpolate_func(inputs):
        idx = torch.from_numpy(np.array([[[0,1,2],[1,2,3]]])).int().cuda()
        weight = torch.from_numpy(np.array([[[1,1,1],[2,2,2]]])).float().cuda()
        interpolated_feats = pointnet2_utils.three_interpolate(inputs, idx, weight)
        return interpolated_feats
    
    assert (gradcheck(interpolate_func, feats, atol=1e-1, rtol=1e-1))

if __name__=='__main__':
    #test_interpolation_grad()
    #xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=False)
    xyz = torch.Tensor([[[ 0.0234, -0.9751,  1.4011],
         [-0.2499,  1.6626, -0.3811],
         [ 0.8417, -0.5145,  0.7864],
         [ 0.2058,  1.1784,  0.3849],
         [ 0.5544, -0.6031, -0.0555],
         [-1.7716,  0.5014, -0.6067],
         [ 2.5301,  0.5504, -0.4435],
         [-1.1078,  0.3823,  0.2507],
         [-0.7827,  0.9522,  0.4905]],

        [[ 0.7612, -1.1062, -1.7496],
         [ 1.1352,  0.5164,  0.7210],
         [ 0.8828,  1.0338, -0.7701],
         [-0.4920,  0.8126, -0.2284],
         [ 0.1904, -2.3758, -0.3887],
         [ 0.0711, -2.1078, -0.5276],
         [ 0.4946,  1.5537,  0.2516],
         [ 0.2828,  0.6082, -1.1267],
         [-1.0092, -2.4192, -0.2919]]]).to('cuda:0')
    inds, dist = pointnet2_utils.fps_light(xyz, 2)
    print("======Point cloud information=====")
    print(xyz)
    print("======Sampling center points distance from avg=====")
    print(inds)
    print(dist)

    xyz_flipped = xyz.transpose(1, 2).contiguous()
    new_xyz = pointnet2_utils.gather_op_cpu(
        xyz_flipped, inds
    ).transpose(1, 2).contiguous()
    print("======Sampling and gathering distance from avg=====")
    print(new_xyz)

    inds2 = pointnet2_utils.furthest_point_sample(xyz, 2)
    new_xyz2 = pointnet2_utils.gather_operation(
        xyz_flipped, inds2
    ).transpose(1, 2).contiguous()
    print("======Sampling and gathering FPS method=====")
    print(inds2)
    print(new_xyz2)

    #Ball query
    inds3 = pointnet2_utils.ball_query_cpu(2, 2, xyz, new_xyz, dist, inds)
    print("======Ball query=====")
    print(inds3)

    #Inverse ball query
    inds4 = pointnet2_utils.inv_ball_query_cpu(xyz, new_xyz, inds3, inds)
    print("======Inverse Ball query=====")
    print(inds4)

    #features = Variable(torch.randn(2, 4, 2).cuda(), requires_grad=False)
    features = torch.Tensor([[[-0.6665,  0.0526],
         [ 0.0153,  1.0042],
         [-0.9345, -0.4614],
         [ 0.3896,  1.3611]],

        [[ 1.6132, -0.4626],
         [ 0.2349, -1.2845],
         [ 0.8065, -0.4818],
         [-0.2079, -2.4726]]]).to('cuda:0')
    print("======Randomly generated features=====")
    print(features)
    
    #Interpolate based on inverse ball query result
    prop_features = pointnet2_utils.inv_interpolate_cpu(features, inds4)
    print("======Propagated features=====")
    print(prop_features)
