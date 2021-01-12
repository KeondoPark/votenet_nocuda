// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

using namespace std;

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data<float>(),
                                 idx.data<int>(), output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data<float>(),
                                      idx.data<int>(), output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);   
  
  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data<float>(),
        tmp.data<float>(), output.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}


//Added
at::Tensor distance_from_avg(at::Tensor points, const int n) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);   

  at::Tensor output =
      torch::zeros({n}, at::device(points.device()).dtype(at::ScalarType::Float));

  const float* dataset = (float *) points.data<float>();
  float* distances = output.data<float>();

  float avgx = 0;
  float avgy = 0;
  float avgz = 0;    

  for (int n_idx = 0; n_idx < n; ++n_idx){
    float x = dataset[n_idx*3 + 0];
    float y = dataset[n_idx*3 + 1];
    float z = dataset[n_idx*3 + 2];

    avgx += x;
    avgy += y;
    avgz += z;
  }

  avgx = avgx / n;
  avgy = avgy / n;
  avgz = avgz / n;   


  for (int n_idx = 0; n_idx < n; ++n_idx){
    float x = dataset[n_idx*3 + 0];
    float y = dataset[n_idx*3 + 1];
    float z = dataset[n_idx*3 + 2];

    distances[n_idx] = (x - avgx) * (x - avgx) + (y - avgy) * (y - avgy) + (z - avgz) * (z - avgz);    
  }
  
  return output;
}

//Added
at::Tensor distance_from_point(at::Tensor points, at::Tensor from_point, const int n) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);   

  at::Tensor output =
      torch::zeros({n}, at::device(points.device()).dtype(at::ScalarType::Float));

  const float* dataset = (float *) points.data<float>();
  const float* pt = (float *) from_point.data<float>();
  float* distances = output.data<float>();

  float pt_x = pt[0];
  float pt_y = pt[1];
  float pt_z = pt[2];

  for (int n_idx = 0; n_idx < n; ++n_idx){
    float x = dataset[n_idx*3 + 0];
    float y = dataset[n_idx*3 + 1];
    float z = dataset[n_idx*3 + 2];

    distances[n_idx] = (x - pt_x) * (x - pt_x) + (y - pt_y) * (y - pt_y) + (z - pt_z) * (z - pt_z);    
  }
  
  return output;
}

/*
//Added
at::Tensor cosine_xyplane(at::Tensor points, const int n) {
  CHECK_CONTIGUOUS(points);  
  CHECK_IS_FLOAT(points);     

  at::Tensor output =
      torch::zeros({n}, at::device(points.device()).dtype(at::ScalarType::Float));

  const float* dataset = (float *) points.data<float>();  
  float* cosine = output.data<float>();

  for (int n_idx = 0; n_idx < n; ++n_idx){
    float x = dataset[n_idx*3 + 0];
    float y = dataset[n_idx*3 + 1];

    cosine[n_idx] = sqrt(x)
  }
  
  return output;
}
*/
/*
at::Tensor furthest_point_sampling_noncuda(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);   
  
  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor batch_distances =
      torch::zeros({points.size(0), points.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  at::Tensor batch_distances2 =
      torch::zeros({points.size(0), points.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));


  int b = points.size(0);
  int n = points.size(1);
  int m = nsamples;

  float* dataset = points.data<float>();
  float* distances = batch_distances.data<float>();
  float* distances2 = batch_distances2.data<float>();

  for (int b_idx = 0; b_idx < b; ++b){
    float avgx = 0;
    float avgy = 0;
    float avgz = 0;    

    for (int n_idx = 0; n_idx < n; n += 3){
      float x = dataset[b_idx * n * 3 + n_idx + 0];
      float y = dataset[b_idx * n * 3 + n_idx + 1];
      float z = dataset[b_idx * n * 3 + n_idx + 2];

      avgx += x;
      avgy += y;
      avgz += z;
    }

    avgx = avgx / n;
    avgy = avgy / n;
    avgz = avgz / n;   

    for (int n_idx = 0; n_idx < n; n += 3){
      float x = dataset[b_idx * n * 3 + n_idx + 0];
      float y = dataset[b_idx * n * 3 + n_idx + 1];
      float z = dataset[b_idx * n * 3 + n_idx + 2];

      distances[b_idx * n + n_idx] = (x - avgx) * (x - avgx) + (y - avgy) * (y - avgy) + (z - avgz) * (z - avgz);
    }

    vector<size_t> x = sort_indexes(distances);
    
    //Farthest point
    float far_x = dataset[b_idx * n  * 3 + x[n-1] + 0];
    float far_y = dataset[b_idx * n  * 3 + x[n-1] + 1];
    float far_z = dataset[b_idx * n  * 3 + x[n-1] + 2];

    for (int n_idx = 0; n_idx < n; n += 3){
      float x = dataset[b_idx * n * 3 + n_idx + 0];
      float y = dataset[b_idx * n * 3 + n_idx + 1];
      float z = dataset[b_idx * n * 3 + n_idx + 2];

      distances2[b_idx * n + n_idx] = (x - far_x) * (x - far_x) + (y - far_y) * (y - far_y) + (z - far_z) * (z - far_z);
    }


    int step = (int) n / m;

    for (int i  = 0; i < m; ++i){
      output.data<float>[b_idx * m + i] = x[i * step];
    }
  }

  return output, batch_distances, batch_distances2;
}

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}*/