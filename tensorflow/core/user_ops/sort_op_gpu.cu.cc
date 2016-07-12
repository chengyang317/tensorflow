#include "sort_op_gpu.h"
#include <array>
#include <stdio.h>


template <typename T>
__global__ void bitonic_sort_step(T *input, int *output, int j, int k)
{
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj)>i) {
        if ((i&k)==0) {
            /* Sort ascending */
            if (input[output[i]] > input[output[ixj]]) {
                /* exchange(i,ixj); */
                float temp = output[i];
                output[i] = output[ixj];
                output[ixj] = temp;
            }
        }
        if ((i&k)!=0) {
            /* Sort descending */
            if (input[output[i]] < input[output[ixj]]) {
                /* exchange(i,ixj); */
                float temp = output[i];
                output[i] = output[ixj];
                output[ixj] = temp;
            }
        }
    }
}



template <typename T>
bool BitonicSortLauncher(const T* input, int* output, const int input_size, const Eigen::GpuDevice& d)
{
    const int threads_per_block = 1024;
    int output_size = input_size;
    int blocks_per_grid = (output_size + threads_per_block - 1) / threads_per_block;

    int* indices = new int [output_size];
    for (int i = 0; i < output_size; i++) indices[i] = i;

    cudaMemcpy(output, indices, output_size, cudaMemcpyHostToDevice);

    int j, k;
    /* Major step */
    for (k = 2; k <= output_size; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            bitonic_sort_step<<<blocks_per_grid, threads_per_block, 0, d.stream()>>>(input, output, j, k);
            if (!d.ok()) return false;
        }
    }
    return true;
}































