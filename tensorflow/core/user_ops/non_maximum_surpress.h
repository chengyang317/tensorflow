//
// Created by Philip Cheng on 7/5/16.
//

#ifndef TENSORFLOW_NON_MAXIMUM_SURPRESS_H
#define TENSORFLOW_NON_MAXIMUM_SURPRESS_H

namespace functor {

    template <typename Device, typename T>
    struct ExtractImagePatchesForward {
        void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                        int patch_rows, int patch_cols, int stride_rows,
                        int stride_cols, int rate_rows, int rate_cols,
                        const Eigen::PaddingType& padding,
                        typename TTypes<T, 4>::Tensor output) {
            // Need to swap row/col when calling Eigen, because our data is in
            // NHWC format while Eigen assumes NWHC format.
            To32Bit(output).device(d) =
                    To32Bit(input)
                            .extract_image_patches(patch_cols, patch_rows, stride_cols,
                                                   stride_rows, rate_cols, rate_rows, padding)
                            .reshape(output.dimensions());
        }
    };













#endif //TENSORFLOW_NON_MAXIMUM_SURPRESS_H
