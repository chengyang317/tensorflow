//
// Created by Philip Cheng on 6/30/16.
//
#include <vector>
#include <iostream>
#include <array>
#include <algorithm>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace std;
using namespace tensorflow;

REGISTER_OP("NonMaximumSurpress")
.Attr("T: {float, double}")
.Attr("thresh: float32")
.Input("boxes: T")
.Output("indices: int32");


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}


template <typename Device, typename T>
class NonMaximumSurpressOp: public OpKernel {
public:
    explicit NonMaximumSurpressOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("thresh", &thresh_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& boxes = context->input(0);
        auto boxes_tensor = boxes.tensor<T, 2>();

        OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));
        OP_REQUIRES(context, boxes.dim_size(1) == 5, errors::InvalidArgument("boxes' second dim must be 5"));

        int boxes_nums = boxes.dim_size(0);
        TensorShape output_shape;
        TensorShapeUtils::MakeShape(&boxes_nums, 1, &output_shape);
        // Create output tensors
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));


        vector<T> scores(boxes_nums);
        vector<array<T, 4>> cords(boxes_nums);
        vector<T> areas(boxes_nums);
        for (int index = 0; index < boxes_nums; index++)
        {
            scores[index] = boxes_tensor(index, 4);
            for (int i = 0; i < 4; i++) {
                cords[index][i] = boxes_tensor(index, i);
            }
            areas[index] = (cords[index][2] - cords[index][0]) * (cords[index][3] - cords[index][1]);
        }

        vector<size_t> sort_indices = sort_indexes(scores);

        vector<int> keep;
        while (sort_indices.size() > 0) {
            int ind = sort_indices.front();
            keep.push_back(ind);
            vector<size_t> new_sort_indices;
            for (auto it = sort_indices.begin(); it != sort_indices.end(); ++it) {
                int sort_indice = *it;
                array<T, 4> cord_array;
                for (int i = 0; i < 4; i++) {
                    if (i < 2) {
                        cord_array[i] = max(cords[sort_indice][i], cords[ind][i]);
                    } else {
                        cord_array[i] = min(cords[sort_indice][i], cords[ind][i]);
                    }
                }
                float inter_area = max(
                        static_cast<T>(0), cord_array[2] - cord_array[0]) * max(static_cast<T>(0),
                                                                                cord_array[3] - cord_array[1]);
                float ratio = inter_area / areas[ind];
                if (ratio <= thresh_) {
                    new_sort_indices.push_back(sort_indice);
                }
            }
            sort_indices = new_sort_indices;
        }
        // construct the output shape
        int keep_size = keep.size();
        TensorShape output_shape;
        TensorShapeUtils::MakeShape(&keep_size, 1, &output_shape);
        // Create output tensors
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_tensor = output->template tensor<int32, 1>();
        for (int i = 0; i != keep_size; i++) {
            output_tensor(i) = keep[i];
        }
    }

private:
    float thresh_;
};


REGISTER_KERNEL_BUILDER(Name("NonMaximumSurpress").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                                                                            NonMaximumSurpressOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("NonMaximumSurpress").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                                                                            NonMaximumSurpressOp<CPUDevice, double >);




















