#include<stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128

extern "C" {
    __host__ void cpu_to_cuda(float** data, int size) {
        float* data_tmp;
        cudaMalloc((void **)&data_tmp, size * sizeof(float));
        cudaMemcpy(data_tmp, *data, size * sizeof(float), cudaMemcpyHostToDevice);
        free(*data);
        *data = data_tmp;
    }

    __host__ void cuda_to_cpu(float** data, int size) {
        float* data_tmp = (float*)malloc(size * sizeof(float));
        cudaMemcpy(data_tmp, *data, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(*data);
        *data = data_tmp;
    }

    __global__ void add_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            result_data[i] = data1[i] + data2[i];
        }
    }

    __host__ void add_tensor_cuda(float* a, float* b, float* result, int size) {
        int number_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        add_tensor_cuda_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(a, b, result, size);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        cudaDeviceSynchronize();
    }
}
