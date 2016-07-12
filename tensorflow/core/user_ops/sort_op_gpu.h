//
// Created by Philip Cheng on 7/6/16.
//


#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_SORT_OP_GPU_H_
#define TENSORFLOW_USER_OPS_SORT_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
    template <typename T>
    bool BitonicSortLauncher(const T* input, int* output, const int input_size, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif  // TENSORFLOW_USER_OPS_SORT_OP_GPU_H_

