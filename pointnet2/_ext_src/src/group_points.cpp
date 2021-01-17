// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "group_points.h"


at::Tensor group_points_nocuda(at::Tensor points, at::Tensor idx) {

  //printf("Strting group_points_nocuda...\n");

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  int B = points.size(0);
  int c = points.size(1);
  int n = points.size(2);

  int m = idx.size(1);
  int nsample = idx.size(2);

  const float* features = (float *) points.data<float>();
  const int* inds = (int *) idx.data<int>();

  float* grouped = output.data<float>();

  for (int b_idx = 0; b_idx < B; ++b_idx){
    for (int m_idx = 0; m_idx < m; ++m_idx){
      for (int ns_idx = 0; ns_idx < nsample; ++ns_idx){
        int index_n = inds[b_idx * m * nsample + m_idx * nsample + ns_idx];
        for (int c_idx = 0; c_idx < c; ++c_idx){
          grouped[b_idx * c * m * nsample + c_idx * m * nsample + m_idx * nsample + ns_idx] = features[b_idx * c * n + c_idx * n + index_n];
        }
      }
    }
  }


  return output;
}

at::Tensor group_points_grad_nocuda(at::Tensor grad_out, at::Tensor idx, const int n) {

  //printf("Strting group_points_grad_nocuda...\n");

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  int B = grad_out.size(0);
  int c = grad_out.size(1);
  int m = grad_out.size(2);
  int nsample = grad_out.size(3);
  
  const float* grad_out_data = (float *) grad_out.data<float>();
  const int* inds = (int *) idx.data<int>();

  float* grad_points = output.data<float>();

  for (int b_idx = 0; b_idx < B; ++b_idx){    
    for (int m_idx = 0; m_idx < m; ++m_idx){
      for (int ns_idx = 0; ns_idx < nsample; ++ns_idx){
        int index_n = inds[b_idx * m * nsample + m_idx * nsample + ns_idx];
        for (int c_idx = 0; c_idx < c; ++c_idx){      
          grad_points[b_idx * c * n + c_idx * n + index_n] += grad_out_data[b_idx * c * m * nsample + c_idx * m * nsample + m_idx * nsample + ns_idx];
        }
      }
    }
  }

  return output;
}