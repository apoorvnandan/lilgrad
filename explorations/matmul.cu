#include <cuda_runtime.h>
#include <stdio.h>

typedef struct {
    float* data;
    float* cuda_data;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" {
    void cpu_to_cuda(Arr* a) {
    	float* cuda_data;
        size_t size_a = a->size * sizeof(float);
        cudaMalloc((void**)&cuda_data, size_a);
        cudaMemcpy(cuda_data, a->data, size_a, cudaMemcpyHostToDevice);
	a->cuda_data = cuda_data;
    }

    void cuda_to_cpu(Arr* a) {
        size_t size_a = a->size * sizeof(float);
   	float* cpu_data = (float*) malloc(size_a);
        cudaMemcpy(cpu_data, a->cuda_data, size_a, cudaMemcpyDeviceToHost);
	cudaFree(a->cuda_data);
	a->data = cpu_data;
    }

    cudaError_t matmul_arr(Arr* c, Arr* a, Arr* b) {
	int M = a->shape[0];
	int K = a->shape[1];
	int N = b->shape[1];
        dim3 threadsPerBlock(32,32);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_kernel<<<numBlocks, threadsPerBlock>>>(a->cuda_data, b->cuda_data, c->cuda_data, M, N, K);
        cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
	cudaDeviceSynchronize();
	return error;
    }
}
