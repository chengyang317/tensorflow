//
// Created by Philip Cheng on 7/6/16.
//


#ifndef TENSORFLOW_USER_OPS_SORT_OP_GPU_H_
#define TENSORFLOW_USER_OPS_SORT_OP_GPU_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

    typedef Eigen::GpuDevice GPUDevice;

    template <typename T>
    bool BitonicSortLauncher(const T* input, int* output, const int input_size, const GpuDevice& d);

}  // namespace tensorflow

#endif  // TENSORFLOW_USER_OPS_SORT_OP_GPU_H_

