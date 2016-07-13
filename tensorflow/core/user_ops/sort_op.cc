//
// Created by Philip Cheng on 7/6/16.
//
#include <vector>
#include <algorithm>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


REGISTER_OP("Sort")
.Attr("T: {float, double, int}")
.Input("input: T")
.Output("indices: int");


using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T>
class SortOp: public OpKernel {
public:
    explicit SortOp(OpKernelConstruction* context): OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor &input = context->input(0);
        OP_REQUIRES(context, input.dims() == 1, errors::InvalidArgument("input must be 1 dimensional"));

        auto input_flat = input.flat<T>();
        int input_size = input_flat.size();

        TensorShape output_shape;
        TensorShapeUtils::MakeShape(&input_size, 1, &output_shape);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_flat = output->template flat<int>();

        std::vector<int> idx(input_size);
        std::vector <T> input_v(input_size);
        for (int i = 0; i != idx.size(); ++i) {
            idx[i] = i;
            input_v[i] = input_flat(i);
        }

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(), [&input_v](int i1, int i2) { return input_v[i1] > input_v[i2]; });

        for (int i = 0; i != idx.size(); ++i) output_flat(i) = idx[i];
    }
};


#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Sort").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SortOp<CPUDevice, type>)

REGISTER_KERNEL(int);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL


#if GOOGLE_CUDA

template <typename T>
bool BitonicSortLauncher(const T* input, int* output, const int input_size, const Eigen::GpuDevice& d);


template <typename T>
static void SortKernel(OpKernelContext* context, const Tensor* input, Tensor* output, const int input_size)
{
    BitonicSortLauncher(
        input->flat<T>().data(), output->flat<int>().data(), input_size, context->eigen_device<Eigen::GpuDevice>());
}




template <typename T>
class SortOp<GPUDevice, T>: public OpKernel {
public:
    typedef GPUDevice Device;
    explicit SortOp(OpKernelConstruction* context): OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor &input = context->input(0);
        OP_REQUIRES(context, input.dims() == 1, errors::InvalidArgument("input must be 1 dimensional"));

        int input_size = input.NumElements();

        TensorShape output_shape;
        TensorShapeUtils::MakeShape(&input_size, 1, &output_shape);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        SortKernel(context, &input, output, input_size);
    }
}


#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Sort").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SortOp<GPUDevice, type>)

REGISTER_KERNEL(int);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

#endif











































