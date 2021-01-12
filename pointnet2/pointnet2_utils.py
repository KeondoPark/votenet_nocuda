# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
import sys
import time
try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        
        fps_inds = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthest_point_sample = FurthestPointSampling.apply


import random
import numpy as np

def distance(p1, p2):    
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) ** 0.5

def distance2(p1, p2):    
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 

class FPS_light(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):    
        
        B = xyz.shape[0]        
        n = xyz.shape[1]        
        for batch_index in range(B):            
            remaining_points = xyz[batch_index]               
            #avg_point = torch.mean(remaining_points, 0)            
            #distances = torch.tensor([distance(p, avg_point) for p in remaining_points])
            
            distances = distance_from_avg(remaining_points.cpu(), n)
            distances = torch.sqrt(distances)
            
            x = torch.argsort(distances)     
            step = int(n/npoint)            

            solution_set = torch.zeros(npoint, dtype=torch.int32, device="cpu")

            #farthest_point = remaining_points[x[-1]]            
            #distances2 = torch.tensor([distance(p, farthest_point) for p in remaining_points])
            
            #distances2 = distance_from_point(remaining_points.cpu(), farthest_point.cpu(), n)
            #distances2 = torch.sqrt(distances2)
            
            
            for i in range(npoint):
                solution_set[i] = x[i * step]

            solution_set = torch.unsqueeze(solution_set, 0)
            distances = torch.unsqueeze(distances, 0)
            #distances2 = torch.unsqueeze(distances2, 0)

            if batch_index == 0:        
                batch_solution_set = solution_set
                batch_distances = distances
                #batch_distances2 = distances2
            else:
                batch_solution_set = torch.cat((batch_solution_set, solution_set), 0)                
                batch_distances = torch.cat((batch_distances, distances), 0)        
                #batch_distances2 = torch.cat((batch_distances2, distances2), 0)        

        return batch_solution_set, batch_distances #, batch_distances2

    @staticmethod
    def backward(xyz, a=None, b=None):
        return None, None
    
fps_light = FPS_light.apply


class DistanceFromAvg(Function):
    @staticmethod
    def forward(ctx, points, n):
        distance = _ext.distance_from_avg(points, n)        
        return distance

    @staticmethod
    def backward(ctx, a=None):
        return None, None

distance_from_avg = DistanceFromAvg.apply

class DistanceFromPoint(Function):
    @staticmethod
    def forward(ctx, points, from_point, n):
        distance = _ext.distance_from_point(points, from_point, n)        
        return distance

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

distance_from_point = DistanceFromPoint.apply


def random_sampling(xyz, npoint):
    """ 
    xyz : torch.Tensor
        (B, N, 3) tensor where N > npoint
    npoint : int32
        number of features in the sampled set
    """
    pc = xyz.cpu().numpy()
    B = pc.shape[0]  
    print('pc shape: ', pc.shape)
    
    for batch_index in range(B):        
        choices = np.random.choice(pc.shape[1], npoint, replace=None)
        
        if batch_index == 0:        
            batch_solution_set = np.expand_dims(choices, 0)
        else:
            choices = np.expand_dims(choices, 0)
            batch_solution_set = np.append(batch_solution_set, choices, axis = 0) 
    
    return torch.from_numpy(batch_solution_set).type(torch.int32).to("cpu")

class gather_cpu(Function):
    @staticmethod
    def forward(ctx, features, idx):

        b = features.shape[0]
        c = features.shape[1]
        n = features.shape[2]
        m = idx.shape[1]        

        out_point = torch.zeros([1, c, 1], dtype = torch.float32, device = "cpu")

        ctx.for_backwards = (idx, b, c, n)
        
        for i in range(b):        
            for k in range(m):            
                j = idx[i, k]
                #out_point = features[i, :, j] 
                if k == 0:                
                    #output_per_batch = torch.unsqueeze(out_point, 1)
                    out_point = features[i, :, j]
                    output_per_batch = torch.unsqueeze(out_point, 1)
                else:
                    out_point = torch.unsqueeze(features[i, :, j], 1)
                    output_per_batch = torch.cat((output_per_batch, out_point), 1)
            if i == 0:
                gather_points = torch.unsqueeze(output_per_batch, 0)
            else:
                output_per_batch = torch.unsqueeze(output_per_batch, 0)
                gather_points = torch.cat((gather_points, output_per_batch), 0)
        
        return gather_points

    @staticmethod
    def backward(ctx, grad_out):
        idx, b, c, n = ctx.for_backwards       
        
        grad_features = torch.zeros([b, c, n], dtype=torch.float32, device = "cpu")
        m = idx.shape[1]

        for b_idx in range(b):
            for m_idx in range(m):
                a = idx[b_idx, m_idx]
                grad_features[b_idx, :, a] = grad_out[b_idx, :, m_idx]

        return grad_features, None

gather_op_cpu = gather_cpu.apply

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)
        #print(_ext.gather_points(features, idx))

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply

class InvBallQuery(Function):
    r"""
    Find the three nearest neighbors of unknown in known
    Parameters
    ----------
    unknown : torch.Tensor
        (B, n, 3) tensor of known features
    known : torch.Tensor
        (B, m, 3) tensor of unknown features
/    grouped_xyz: torch.Tensor
        (B, m, k): tensor of points ball-queried around center points
    inds: torch.Tensor
        (B, m): Information of center points
    
    Returns
    -------
    idx : torch.Tensor
        (B, n, cp) index of Centers of ball query
    """
    @staticmethod
    def forward(ctx, unknown, known, grouped_xyz, inds):
        b = unknown.shape[0]
        n = unknown.shape[1]
        m = known.shape[1]

        centers_per_point = int(n / m)
        
        centers = torch.zeros([b, n, centers_per_point], dtype = torch.int32, device = "cpu")

        for b_idx in range(b):
            #For each point in Unkown, check whether the point is in the ball query of each center
            for n_idx in range(n):
                j = 0                
                # For the centerpoint, query results are filled with itself repeated
                for m_idx in range(m):                        
                    if inds[b_idx, m_idx] == n_idx:
                        while j < centers_per_point:
                            centers[b_idx, n_idx, j] = m_idx
                            j += 1
                        break                
                if j >= centers_per_point: continue

                # For points not center points
                for m_idx in range(m):                        
                    if n_idx in grouped_xyz[b_idx, m_idx]:
                        centers[b_idx, n_idx, j] = m_idx
                        j += 1
                    if j >= centers_per_point:
                        break
                uniq_cnt = j
                #If no center point is close enough to the point, this point is isolated and not very informative.
                #Maybe better to ignore?
                if j == 0:
                    centers[b_idx, n_idx, :] = torch.randint(0, m, (centers_per_point,), dtype=torch.int32)
                    #print("Isolated point")
                else:
                    while j < centers_per_point:
                        centers[b_idx, n_idx, j] = centers[b_idx, n_idx, j - uniq_cnt]
                        j += 1
        return centers
    
    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

inv_ball_query_cpu = InvBallQuery.apply         


class InvBallQuery_nocuda(Function):
    @staticmethod
    def forward(ctx, unknown, known, grouped_xyz, inds):
        centers = _ext.inv_ball_query_nocuda(unknown.cpu(), known.cpu(), grouped_xyz.cpu(), inds.cpu())        
        return centers

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, N

inv_ball_query_nocuda = InvBallQuery_nocuda.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply

class InvInterpolate_cpu(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, cp) three nearest neighbors of the target features in features        

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        _, n, cp = idx.size()        

        ctx.three_interpolate_for_backward = (idx, m)

        out = torch.zeros([B, c, n], dtype = torch.float32, device = "cpu")        

        for b_idx in range(B):
            for n_idx in range(n):                
                for cp_idx in range(cp):
                    w  = idx[b_idx, n_idx, cp_idx]
                    for c_idx in range(c):                        
                        out[b_idx, c_idx, n_idx] += features[b_idx, c_idx, w]
                    
        out = out / cp

        return out

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        ctx.three_interpolate_for_backward = (idx, m)

        B, c, n = grad_out.size()
        _, _, cp = idx.size()

        grad_features = torch.zeros([B, c, m], dtype = torch.float32, device = "cpu")

        for b_idx in range(B):
            for n_idx in range(n):
                for j in range(cp):
                    m_idx = idx[b_idx, n_idx, j]
                    for c_idx in range(c):
                        grad_features[b_idx, c_idx, m_idx] += grad_out[b_idx, c_idx, n_idx] / cp

        return grad_features, None

inv_interpolate_cpu = InvInterpolate_cpu.apply

class InvInterpolate_nocuda(Function):
    """
        Performs inverse interpolation based on Inverse ball query result
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, cp) Inverse ball query result of the target features in features        

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
    @staticmethod
    def forward(ctx, features, idx):     
        _, _, m = features.size()
        ctx.three_interpolate_for_backward = (idx, m)
        features = features.transpose(1, 2).contiguous()
        output = _ext.inv_interpolate_nocuda(features.cpu(), idx.cpu())        
        output = output.transpose(1, 2).contiguous()       

        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, m = ctx.three_interpolate_for_backward

        grad_out = grad_out.transpose(1,2).contiguous()
        grad_features = _ext.inv_interpolate_nocuda_grad(grad_out.cpu(), idx.cpu(), m)        

        return grad_features, None

inv_interpolate_nocuda = InvInterpolate_nocuda.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class GroupingOperation_nocuda(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points_nocuda(features.cpu(), idx.cpu())

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad_nocuda(grad_out.contiguous().cpu(), idx.cpu(), N)

        return grad_features, None


grouping_operation_nocuda = GroupingOperation_nocuda.apply


class BallQuery_cpu(Function):
    """
    Parameters
    ----------
    radius : float
        radius of the balls
    nsample : int
        maximum number of features in the balls
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the ball query
    batch_distances: torch.Tensor
        (B, m): distances of center points from Average point
    inds: torch.Tensor
        (B, m): Information of center points

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz, batch_distances, inds):
        def filter_func(x, d, r):
            return x > d - r and d < d + r

        b = xyz.shape[0]
        n = xyz.shape[1]
        m = new_xyz.shape[1]            
        r2 = radius ** 2

        for b_idx in range(b):
            for m_idx in range(m):
                if m_idx % 100 == 0:
                    print(m_idx)
                center = new_xyz[b_idx, m_idx]
                # find the distance of center point
                d = batch_distances[b_idx, inds[b_idx, m_idx]].item()
                #d2 = batch_distances2[b_idx, inds[b_idx, m_idx]].item()

                # find candidate points whose distances are between d-r and d+r          
                candidates = torch.zeros(nsample, dtype = torch.int32, device = "cpu")

                j = 0
                
                #for i in range(n):   
                    #May or may not include the center itself 
                    #if i == inds[b_idx, m_idx]: continue 
                    #if filter_func(batch_distances[b_idx, i], d, radius):                        
                        #Check if candidate point is really within radius distance
                        #if distance2(xyz[b_idx, i], center) < r2:                    
                            #candidates[j] = i
                            #j += 1   
                            #if j == nsample: break
                
                tmp = batch_distances[b_idx].cpu().numpy()
                #tmp2 = batch_distances2[b_idx].cpu().numpy()
                
                filtered = (tmp >= d - radius) & (tmp <= d + radius)
                #filtered2 = filtered & (tmp2 >= d2 - radius) & (tmp2 <= d2 + radius)
                #print("filtered:", sum(filtered))
                #print("filtered2:", sum(filtered2))

                cnt = 0
                for i, f in enumerate(filtered):
                    if f:
                        cnt += 1
                        if distance2(xyz[b_idx, i], center) < r2:                    
                            candidates[j] = i
                            j += 1   

                            if j == nsample: 
                                #print(cnt)
                                break                
                        

                while len(candidates) < nsample:            
                    candidates[j] = candidates[j-1]
            
                if m_idx == 0:                    
                    solution_set = torch.unsqueeze(candidates, 0)
                    print(solution_set[0])
                else:
                    solution_set = torch.cat((solution_set, torch.unsqueeze(candidates, 0)), 0)
            
            if b_idx == 0:
                b_solution_set = torch.unsqueeze(solution_set, 0)
            else:
                b_solution_set = torch.cat((b_solution_set, torch.unsqueeze(solution_set, 0)), 0)

        return b_solution_set
    
    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None

ball_query_cpu = BallQuery_cpu.apply



class BallQuery_cpp(Function):
    """
    Parameters
    ----------
    radius : float
        radius of the balls
    nsample : int
        maximum number of features in the balls
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the ball query
    batch_distances: torch.Tensor
        (B, m): distances of center points from Average point
    inds: torch.Tensor
        (B, m): Information of center points

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz, batch_distances, inds):
        def filter_func(x, d, r):
            return x > d - r and d < d + r

        b = xyz.shape[0]
        n = xyz.shape[1]
        m = new_xyz.shape[1]            
        r2 = radius ** 2

        for b_idx in range(b):
            #For cpp extension
            start = time.time()
            x = torch.argsort(batch_distances[b_idx].cpu()).int()
            #x2 = torch.argsort(batch_distances2[b_idx].cpu()).int()

            batch_d = torch.sort(batch_distances[b_idx].cpu())
            #batch_d2 = torch.sort(batch_distances2[b_idx].cpu())
            end = time.time()
            print("Time for sorting before ball query:", end - start)

            solution_set = ball_query_nocuda(new_xyz[b_idx].cpu(), xyz[b_idx].cpu(), radius, nsample, batch_d.values, inds.cpu(), x)
            
            if b_idx == 0:
                b_solution_set = torch.unsqueeze(solution_set, 0)
            else:
                b_solution_set = torch.cat((b_solution_set, torch.unsqueeze(solution_set, 0)), 0)

        return b_solution_set
    
    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None

ball_query_cpp = BallQuery_cpp.apply

class BallQueryNoCuda(Function):
    @staticmethod
    def forward(ctx, new_xyz, xyz, radius, nsample, batch_distances, inds, arg_sort):
        idx = _ext.ball_query_nocuda(new_xyz, xyz, radius, nsample, batch_distances, inds, arg_sort)        
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None, None, None, None

ball_query_nocuda = BallQueryNoCuda.apply

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, batch_distances, inds, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        start = time.time()
        idx = ball_query_cpp(self.radius, self.nsample, xyz, new_xyz, batch_distances, inds)
        end = time.time()
        print("Runtime for Ball query modified cpp: ", end - start)
        print("Ball query cpp result:", idx)
        """
        start = time.time()
        idx = ball_query_cpu(self.radius, self.nsample, xyz, new_xyz, batch_distances, batch_distances2, inds)
        end = time.time()
        print("Runtime for Ball query modified python: ", end - start)
        print("Ball query python result:", idx)        
        """

        #start = time.time()
        #idx = ball_query(self.radius, self.nsample, xyz, new_xyz)        
        #end = time.time()
        #print("Runtime for Ball query original: ", end - start)
        #print("Ball query original result:", idx)

        # During ball query, when there are not enough points around the center
        # the ball is filled with the first point.
        # Below code randomly distributes unique points in such balls
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        #grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz = grouping_operation_nocuda(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            #start = time.time()
            #grouped_features = grouping_operation(features, idx)
            #end = time.time()
            #print("Runtime for grouping_operation original: ", end - start)
            #print(grouped_features)

            start = time.time()
            grouped_features = grouping_operation_nocuda(features, idx)
            end = time.time()
            print("Runtime for grouping_operation NOCUDA: ", end - start)           
            print(grouped_features)


            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        #Keondo: Added for later use
        ret.append(idx)

        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features
