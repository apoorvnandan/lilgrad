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

    __global__ void matmul_kernel(
            float* a, int* shape_a,
            float* b, int* shape_b,
            float* result,
    ) {
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < shape_a[0] && y < shape_b[1]) {
            float tmp = 0.0f;
            for (int i = 0; i < shape_a[1]; i++) {
                tmp += a[x * shape_a[1] + y] * b[x * shape_b[1] + y];
            }
            result[x * shape_b[1] + y] = tmp;
        }
    }

    __host__ void matmul(
            float* a, int* shape_a, 
            float* b, int* shape_b, 
            float* result
    ) {
        int number_of_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        matmul_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(a, shape_a, b, shape_b, result);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        cudaDeviceSynchronize(); 
    }
}
