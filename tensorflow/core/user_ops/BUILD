load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)

tf_custom_op_library(
    name = "sort_op.so",
    srcs = ["sort_op.cc"],
    gpu_srcs = ["sort_op_gpu.cu.cc", "sort_op_gpu.h"],
)

tf_custom_op_library(
    name = "non_maximum_surpress.so",
    srcs = ["non_maximum_surpress.cc"],
)

tf_custom_op_library(
    name = "roi_pooling.so",
    srcs = ["roi_pooling_op.cc"],
    gpu_srcs = ["roi_pooling_op_gpu.cu.cc", "roi_pooling_op_gpu.h"],
)