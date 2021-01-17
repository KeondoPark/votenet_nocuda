// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "interpolate.h"
#include <cstdlib>
#include <experimental/random>

/* Deprecated: Use transposed version below
//Added
at::Tensor inv_interpolate_nocuda(at::Tensor points, at::Tensor idx) {
  
  printf("Starting...");

  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);  
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  

  int B = points.size(0);
  int c = points.size(1);
  int m = points.size(2);
  int n = idx.size(1);
  int cp = idx.size(2);
  

  at::Tensor output =
      torch::zeros({B, c, n},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  const float* features = (float *) points.data<float>();
  const int* inds = (int *) idx.data<int>();

  float* interpolated = output.data<float>();

  printf("B: %d, c: %d, m: %d, n: %d, cp: %d\n", B, c, m, n, cp);
  printf("inds[0]: %d\n", inds[0]);

  for(int b_idx = 0; b_idx < B; ++b_idx){
    for (int n_idx = 0; n_idx < n; ++n_idx){
      for (int cp_idx = 0; cp_idx < cp; ++cp_idx){
        //printf("b_idx * n * cp + n_idx * cp + cp_idx: %d\n", b_idx * n * cp + n_idx * cp + cp_idx);
        int w = inds[b_idx * n * cp + n_idx * cp + cp_idx];        
        //printf("w: %d\n", w);
        for (int c_idx = 0; c_idx < c; ++c_idx){
          interpolated[b_idx * c * n + c_idx * n + n_idx] += features[b_idx * c * m + c_idx * m + w];
        }                 
      }
      //printf("n: %d done\n", n_idx);
    }
  }

  printf("Interpolation done\n");

  //Devide by the number of center points(Averaging)
  for (int i = 0; i < B * c * n; ++i){
    interpolated[i] = interpolated[i] / cp;
  }
  printf("Averaging done\n");
 
  return output;
}
*/

//Input and output is transposed for increasing cache utility
// (B, c, m) -> (B, m, c)
// (B, c, n) -> (B, c, n)
at::Tensor inv_interpolate_nocuda(at::Tensor points, at::Tensor idx) {

  //printf("Strting inv_interpolate_nocuda...\n");
  
  int B = points.size(0);  
  int m = points.size(1);
  int c = points.size(2);
  int n = idx.size(1);
  int cp = idx.size(2);
  

  at::Tensor output =
      torch::zeros({B, n, c},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  const float* features = (float *) points.data<float>();
  const int* inds = (int *) idx.data<int>();

  float* interpolated = output.data<float>();
  float weight = 1 / cp;

  for(int b_idx = 0; b_idx < B; ++b_idx){
    for (int n_idx = 0; n_idx < n; ++n_idx){
      for (int cp_idx = 0; cp_idx < cp; ++cp_idx){
        int center = inds[b_idx * n * cp + n_idx * cp + cp_idx];        
        for (int c_idx = 0; c_idx < c; ++c_idx){
          interpolated[b_idx * c * n + n_idx * c + c_idx] += features[b_idx * c * m + center * c + c_idx] * weight;
        }         
      }
    }
  }

  //Devide by the number of center points(Averaging)
  for (int i = 0; i < B * c  * n; ++i){
    interpolated[i] = interpolated[i] / cp;
  }
 
  return output;
}

//Input and output is transposed
//Input: (B, n, c)
//Output: (B, m, c)
//Index(Inverse ball query result): (B, n, cp)
at::Tensor inv_interpolate_nocuda_grad(at::Tensor grad_out, at::Tensor idx, int m) {
  
  //printf("Strting inv_interpolate_nocuda_grad...\n");
  
  int B = grad_out.size(0);  
  int n = grad_out.size(1);
  int c = grad_out.size(2);  
  int cp = idx.size(2);
  

  at::Tensor output =
      torch::zeros({B, m, c},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  const float* grad = (float *) grad_out.data<float>();
  const int* inds = (int *) idx.data<int>();

  float* inv_intp = output.data<float>();
  float weight = 1 / cp;

  for (int b_idx = 0; b_idx < B; ++b_idx){
    for (int n_idx = 0; n_idx < n; ++n_idx){
      for (int cp_idx = 0; cp_idx < cp; ++cp_idx){
        int center = inds[b_idx * n * cp + n_idx * cp + cp_idx];
        for (int c_idx = 0; c_idx < c; ++c_idx){
          inv_intp[b_idx * c * m + center * c + c_idx] += grad[b_idx * c * n + n_idx * c + c_idx] * weight;
        }
      }
    }
  }
 
  return output;
}



//Added
at::Tensor inv_ball_query_nocuda(at::Tensor unknowns, at::Tensor knows, at::Tensor grouped_xyz, at::Tensor idx) {

  //printf("Strting inv_ball_query_nocuda...\n");

  int b = unknowns.size(0);
  int n = unknowns.size(1);
  int m = knows.size(1);
  int k = grouped_xyz.size(2);
  int cp = (int) (n / m);

  at::Tensor inv_idx =
      torch::zeros({b, n, cp},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));

  const float* unknown_points = (float *) unknowns.data<float>();
  const float* known_points = (float *) knows.data<float>();
  const int* balls = (int *) grouped_xyz.data<int>();
  const int* inds = (int  *) idx.data<int>();

  int* centers = inv_idx.data<int>();

  for (int b_idx = 0; b_idx < b; ++b_idx){
    for (int n_idx = 0; n_idx < n; ++n_idx){
      int j = 0;
      //Check if n_idx is center point
      for (int m_idx = 0; m_idx < m; ++m_idx){
        if (inds[b_idx * m + m_idx] == n_idx){
          while (j < cp){
            centers[b_idx * n * cp + n_idx * cp + j] = m_idx;
            j++;
          }
          break;
        }
      }

      if (j >= cp)
        continue;

      //if n_idx is not the center point
      for (int m_idx = 0; m_idx < m; ++m_idx){
        for (int k_idx = 0; k_idx < k; ++k_idx){
          if (balls[b_idx * m * k + m_idx * k + k_idx] == n_idx){
            centers[b_idx * n * cp + n_idx * cp + j] = m_idx;
            j++;
            break;
          }
        }
        if (j >= cp)
          break;
      }

      int uniq_cnt = j;

      //If no center point is found, randomly assign center point
      if (j == 0){
        //printf("No center point around. b_idx: %d, n_idx: %d\n", b_idx, n_idx);
        float min_dist = 1e+10;
        int min_m = -1;
        float x = unknown_points[b_idx * n * 3 + n_idx * 3 + 0];
        float y = unknown_points[b_idx * n * 3 + n_idx * 3 + 1];
        float z = unknown_points[b_idx * n * 3 + n_idx * 3 + 2];
        for (int m_idx = 0; m_idx < m; ++m_idx){
          float center_x = known_points[b_idx * m * 3 + m_idx * 3 + 0];
          float center_y = known_points[b_idx * m * 3 + m_idx * 3 + 1];
          float center_z = known_points[b_idx * m * 3 + m_idx * 3 + 2];

          float dist = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y) + (z - center_z) * (z - center_z);

          if (min_dist > dist){
            min_dist = dist;
            min_m = m_idx;
          } 
        }


        while (j < cp){
            //centers[b_idx * n * cp + n_idx * cp + j] = std::experimental::randint(0, m);
            centers[b_idx * n * cp + n_idx * cp + j] = min_m;
            j++;
          }
      } else {
        //if the number of centers found is less than cp, repeat the centers already found
        while (j < cp){
          centers[b_idx * n * cp + n_idx * cp + j] = centers[b_idx * n * cp + n_idx * cp + j - uniq_cnt];
          j++;
        }
      }
    }
  }

  return inv_idx;
}