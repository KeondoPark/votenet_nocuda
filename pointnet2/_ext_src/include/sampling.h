// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

at::Tensor gather_points(at::Tensor points, at::Tensor idx);
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);
at::Tensor distance_from_avg(at::Tensor points, const int n);
at::Tensor distance_from_point(at::Tensor points, at::Tensor from_point, const int n);

