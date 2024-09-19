#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCKSIZE 128

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

__global__ void matmul_transpose(float* A, float* B, float* C, int M, int N, int K) {
    // A is (M,K), B is (N,K), we are calculating A @ B.T which is (M,N)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[col * K + i]; // (i,col) in B.T = (col, i) in B
        }
        C[row * N + col] = sum;
    }
}

__global__ void transpose_matmul(float* A, float* B, float* C, int M, int N, int K) {
    // A is (K,M), B is (K,N), we are calculating A.T @ B which is (M,N)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[i * M + row] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void relu_kernel(float* out, float* inp, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (inp[i] > 0) ? inp[i] : 0.0f;
    }
}

__global__ void relu_backward_kernel(float* dinp, float* dout, float* out, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dinp[i] = (out[i] > 0) ? dout[i] : 0.0f;
    }
}

__global__ void logsoftmax_kernel(float* out, float* inp, int B, int C, int strides_0, int strides_1) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float maxv = -INFINITY;
        for (int c = 0; c < C; c++) {
            int pos = b * strides_0 + c * strides_1;
            maxv = fmax(maxv, inp[pos]);
        }
        float sumexp = 0.0f;
        for (int c = 0; c < C; c++) {
            int pos = b * strides_0 + c * strides_1;
            float expval = expf(inp[pos] - maxv);
            sumexp += expval;
        }
        for (int c = 0; c < C; c++) {
            int pos = b * strides_0 + c * strides_1;
            out[pos] = inp[pos] - maxv - logf(sumexp);
        }
    }
}

__global__ void logsoftmax_backward_kernel(float* dinp, float* dout, float* out, int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    float gradsum = 0.0f;
    for (int c = 0; c < C; c++) {
        gradsum += dout[b * C + c];
    }
    for (int c = 0; c < C; c++) {
        int pos = b*C+c;
        dinp[pos] = dout[pos] - expf(out[pos]) * gradsum;
    }
}

__global__ void loss_kernel(float* loss, float* outs, float* labels, size_t n) {
    __shared__ float sharedSum;
    if (threadIdx.x == 0) sharedSum = 0.0f;
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += outs[i] * labels[i];
    }
    atomicAdd(&sharedSum, sum);
    __syncthreads();
    if (threadIdx.x == 0) {
    loss[0] = sharedSum / n;
    }
}

__global__ void loss_backward_kernel(float* douts, float* labels, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        douts[idx] = labels[idx];
    }
}

__global__ void update_weight_kernel(float* w, float* dw, float lr, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = w[i] - dw[i] * lr;
    }
}

cudaError_t checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaDeviceSynchronize();
    return error;
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

    cudaError_t matmul(Arr* c, Arr* a, Arr* b) {
        int M = a->shape[0];
        int K = a->shape[1];
        int N = b->shape[1];
        dim3 threadsPerBlock(32,32);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_kernel<<<numBlocks, threadsPerBlock>>>(a->cuda_data, b->cuda_data, c->cuda_data, M, N, K);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t matmul_backward_a(Arr* da, Arr* dc, Arr* b) {
        int P = da->shape[0];
        int Q = da->shape[1];
        int R = b->shape[1];
        dim3 threadsPerBlock(32,32);
        dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (Q + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matmul_transpose<<<numBlocks, threadsPerBlock>>>(dc->cuda_data, b->cuda_data, da->cuda_data, P, Q, R);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t matmul_backward_b(Arr* db, Arr* dc, Arr* a) {
        int P = a->shape[0];
        int Q = a->shape[1];
        int R = db->shape[1];
        dim3 threadsPerBlock(32,32);
        dim3 numBlocks((Q + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (R + threadsPerBlock.y - 1) / threadsPerBlock.y);
        transpose_matmul<<<numBlocks, threadsPerBlock>>>(a->cuda_data, dc->cuda_data, db->cuda_data, Q, R, P);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t relu(Arr* out, Arr* inp) {
        size_t n = inp->size;
        int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
        relu_kernel<<<numBlocks, BLOCKSIZE>>>(out->cuda_data, inp->cuda_data, n);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t relu_backward(Arr* dinp, Arr* dout, Arr* out) {
        size_t n = out->size;
        int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
        relu_backward_kernel<<<numBlocks, BLOCKSIZE>>>(dinp->cuda_data, dout->cuda_data, out->cuda_data, n);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t logsoftmax(Arr *out, Arr* inp) {
        int B = inp->shape[0];
        int numBlocks = (B + BLOCKSIZE - 1) / BLOCKSIZE;
        logsoftmax_kernel<<<numBlocks, BLOCKSIZE>>>(out->cuda_data, inp->cuda_data, inp->shape[0], inp->shape[1], inp->strides[0], inp->strides[1]);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t logsoftmax_backward(Arr* dinp, Arr* dout, Arr* out) {
        int B = out->shape[0];
        int C = out->shape[1];
        int numBlocks = (B + BLOCKSIZE - 1) / BLOCKSIZE;
        logsoftmax_backward_kernel<<<numBlocks, BLOCKSIZE>>>(dinp->cuda_data, dout->cuda_data, out->cuda_data, B, C);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t lossfn(Arr* loss, Arr* outs, Arr* labels) {
        size_t n = outs->size;
        int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
        loss_kernel<<<1, BLOCKSIZE>>>(loss->cuda_data, outs->cuda_data, labels->cuda_data, n);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t loss_backward(Arr* douts, Arr* labels) {
        size_t n = douts->size;
        int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
        loss_backward_kernel<<<numBlocks, BLOCKSIZE>>>(douts->cuda_data, labels->cuda_data, n);
        return checkCudaError(cudaGetLastError());
    }

    cudaError_t update_weight(Arr* w, Arr* dw, float lr) {
        size_t n = w->size;
        int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
        update_weight_kernel<<<numBlocks, BLOCKSIZE>>>(w->cuda_data, dw->cuda_data, lr, n);
        return checkCudaError(cudaGetLastError());
    }

    void train_batch(Arr* inp, Arr* labels, Arr* w1, Arr* w2, float lr,
        Arr* w1_out, Arr* relu_out, Arr* w2_out, Arr* logsoftmax_out,
        Arr* loss, Arr* logsoftmax_out_grad, Arr* w2_out_grad,
        Arr* w2_grad, Arr* relu_out_grad, Arr* w1_out_grad, Arr* w1_grad
    ) {
        // FFN training step: inp is (B,C), w1 is (C,D), w2 is (D,E)
        // labels is (B, E) (one hot encoded)
        matmul(w1_out, inp, w1);
        relu(relu_out, w1_out);
        matmul(w2_out, relu_out, w2);
        logsoftmax(logsoftmax_out, w2_out);
        lossfn(loss, logsoftmax_out, labels);
        loss_backward(logsoftmax_out_grad, labels);
        logsoftmax_backward(w2_out_grad, logsoftmax_out_grad, logsoftmax_out);
        matmul_backward_a(relu_out_grad, w2_out_grad, w2);
        matmul_backward_b(w2_grad, w2_out_grad, relu_out);
        relu_backward(w1_out_grad, relu_out_grad, w1_out);
        matmul_backward_b(w1_grad, w1_out_grad, inp);
        update_weight(w1, w1_grad, lr);
        update_weight(w2, w2_grad, lr);
    }
}