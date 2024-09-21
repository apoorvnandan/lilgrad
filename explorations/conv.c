#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* data;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;

Arr* create_arr(int *shape, int ndim);
Arr* create_arr_like(Arr* a);
Arr* copy(Arr* inp);
void free_arr(Arr* a);

void matmul(Arr* c, Arr* a, Arr* b);
void matmul_backward(Arr* da, Arr* db, Arr* dc, Arr* a, Arr* b);
void relu(Arr* out, Arr* inp);
void relu_backward(Arr* dinp, Arr* dout, Arr* out);
void logsoftmax(Arr* out, Arr* inp);
void logsoftmax_backward(Arr* dinp, Arr* dout, Arr* out);
void lossfn(Arr* loss, Arr* a, Arr* b);
void lossfn_backward(Arr* da, Arr* a, Arr* b);
void conv2d(Arr* out, Arr* inp, Arr* kernel);
void maxpool2d(Arr* out, Arr* inp, int kernel_size, int stride);
void flatten(Arr* out, Arr* inp);
void conv2d_backward(Arr* dinp, Arr* dkernel, Arr* dout, Arr* inp, Arr* kernel);
void maxpool2d_backward(Arr* dinp, Arr* dout, Arr* inp, int kernel_size, int stride);

void train_batch(
        Arr* inp, 
        Arr* labels,
        Arr* w1,
        Arr* w2,
        float lr,
        Arr* w1_out, Arr* relu_out, Arr* w2_out,
        Arr* logsoftmax_out, Arr* loss,
        Arr* logsoftmax_out_grad,
        Arr* w2_out_grad, Arr* w2_grad,
        Arr* relu_out_grad, Arr* w1_out_grad,
        Arr* w1_grad
) {
    // FFN training step: inp is (B,C), w1 is (C,D), w2 is (D,E)
    // labels is (B, E) (one hot encoded)
    matmul(w1_out, inp, w1);
    relu(relu_out, w1_out);
    matmul(w2_out, relu_out, w2);
    logsoftmax(logsoftmax_out, w2_out);
    lossfn(loss, logsoftmax_out, labels);
    lossfn_backward(logsoftmax_out_grad, logsoftmax_out, labels);
    logsoftmax_backward(w2_out_grad, logsoftmax_out_grad, logsoftmax_out);
    matmul_backward(
        relu_out_grad, w2_grad, w2_out_grad, relu_out, w2
    );
    relu_backward(w1_out_grad, relu_out_grad, w1_out);
    matmul_backward(
        NULL, w1_grad, w1_out_grad, inp, w1
    );
    // update weights and zero grads
    for (int i = 0; i < w1->size; i++) {
        w1->data[i] = w1->data[i] - lr * w1_grad->data[i];
        w1_grad->data[i] = 0.0f;
    }
    for (int i = 0; i < w2->size; i++) {
        w2->data[i] = w2->data[i] - lr * w2_grad->data[i];
        w2_grad->data[i] = 0.0f;
    }
    memset(logsoftmax_out_grad, 0, logsoftmax_out->size * sizeof(float));
    memset(w2_out_grad, 0, w2_out->size * sizeof(float));
    memset(relu_out_grad, 0, relu_out->size * sizeof(float));
    memset(w1_out_grad, 0, w1_out->size * sizeof(float));
}



void matmul(Arr* c, Arr* a, Arr* b) {
    // (P,Q) x (Q,R) = (P,R)
    int P = a->shape[0];
    int Q = a->shape[1];
    int R = b->shape[1];
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < Q; k++) {
                int pos_a = i * a->strides[0] + k * a->strides[1];
                int pos_b = k * b->strides[0] + j * b->strides[1];
                tmp += a->data[pos_a] * b->data[pos_b];
            }
            int pos_c = i * c->strides[0] + j * c->strides[1];
            c->data[pos_c] = tmp;
        }
    }
}

void matmul_backward(Arr* da, Arr* db, Arr* dc, Arr* a, Arr* b) {
    // a (P,Q), b (Q,R), c (P, R)
    int P = a->shape[0];
    int Q = a->shape[1];
    int R = b->shape[1];
    if (da != NULL) {
        // dc x b.T  (P,R) x (R,Q) => (P,Q)
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < Q; j++) {
                float tmp = 0.0f;
                for (int k = 0; k < R; k++) {
                    // (k,j) in b.T is (j,k) in b
                    int pos_b = j * b->strides[0] + k * b->strides[1]; 
                    tmp += dc->data[i * R + k] * b->data[pos_b];
                }
                int pos_da = i * Q + j;
                da->data[pos_da] = tmp;
            }
        }
    }
    if (db != NULL) {
        // a.T x dc  (Q,P) x (P,R) => (Q,R)
        for (int i = 0; i < Q; i++) {
            for (int j = 0; j < R; j++) {
                float tmp = 0.0f;
                for (int k = 0; k < P; k++) {
                    // (i,k) in a.T is (k,i) in a
                    int pos_a = k * a->strides[0] + i * a->strides[1]; 
                    tmp += dc->data[k * R + j] * a->data[pos_a];
                }
                int pos_db = i * R + j;
                db->data[pos_db] = tmp;
            }
        }
    }
}

void relu(Arr* out, Arr* inp) {
    for (int i = 0; i < inp->size; i++) {
        if (inp->data[i] > 0) {
            out->data[i] = inp->data[i];
        } else {
            out->data[i] = 0.0f;
        }
    }
}

void relu_backward(Arr* dinp, Arr* dout, Arr* out) {
    for (int i = 0; i < out->size; i++) {
        if (out->data[i] > 0) {
            dinp->data[i] = dout->data[i];
        } else {
            dinp->data[i] = 0.0f;
        }
    }
}


void logsoftmax(Arr* out, Arr* inp) {
    // inp and out are both (B,C)
    for (int b = 0; b < inp->shape[0]; b++) {
        float maxv = -INFINITY;
        for (int c = 0; c < inp->shape[1]; c++) {
            int pos = b * inp->strides[0] + c * inp->strides[1];
            maxv = fmax(maxv, inp->data[pos]);            
        }
        float sumexp = 0.0f;
        for (int c = 0; c < inp->shape[1]; c++) {
            int pos = b * inp->strides[0] + c * inp->strides[1];
            float expval = expf(inp->data[pos] - maxv);
            sumexp += expval;
        }
        for (int c = 0; c < inp->shape[1]; c++) {
            int pos = b * inp->strides[0] + c * inp->strides[1];
            out->data[pos] = inp->data[pos] - maxv - logf(sumexp);
        }
    }
}

void logsoftmax_backward(Arr* dinp, Arr* dout, Arr* out) {
    // dout and dinp are both (B,C)
    for (int b = 0; b < out->shape[0]; b++) {
        float gradsum = 0.0f;
        for (int c = 0; c < out->shape[1]; c++) {
            gradsum += dout->data[b * out->shape[1] + c];
        }
        for (int c = 0; c < out->shape[1]; c++) {
            int pos = b*out->shape[1]+c;
            dinp->data[pos] = dout->data[pos] - expf(out->data[pos]) * gradsum;
        }
    }
}


void lossfn(Arr* loss, Arr* a, Arr* b) {
    float s = 0.0f;
    for (int i = 0; i < a->size; i++) {
        s += a->data[i] * b->data[i];    
    }
    loss->data[0] = s/a->size;
}

void lossfn_backward(Arr* da, Arr* a, Arr* b) {
    for (int i = 0; i < a->size; i++) {
        da->data[i] = b->data[i];
    }
}

Arr* create_arr(int *shape, int ndim) {
    Arr* arr = (Arr*)malloc(sizeof(Arr));
    if (!arr) return NULL;

    arr->ndim = ndim;
    arr->shape = (int*)malloc(ndim * sizeof(int));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }
    memcpy(arr->shape, shape, ndim * sizeof(int));

    arr->strides = (int*)malloc(ndim * sizeof(int));
    if (!arr->strides) {
        free(arr->shape);
        free(arr);
        return NULL;
    }

    arr->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        arr->strides[i] = arr->size;
        arr->size *= shape[i];
    }

    arr->data = (float*)calloc(arr->size, sizeof(float));
    if (!arr->data) {
        free(arr->strides);
        free(arr->shape);
        free(arr);
        return NULL;
    }

    return arr;
}

Arr* create_arr_like(Arr* a) {
    if (!a) return NULL;
    return create_arr(a->shape, a->ndim);
}

void free_arr(Arr* a) {
    if (a) {
        free(a->data);
        free(a->shape);
        free(a->strides);
        free(a);
    }
}

Arr* create_randn_arr(int *shape, int ndim, float mean, float std_dev) {
    Arr* arr = create_arr(shape, ndim);
    if (!arr) return NULL;

    // Seed the random number generator
    static int seeded = 0;
    if (!seeded) {
        srand(0);
        seeded = 1;
    }

    // Fill the array with random normal numbers using Box-Muller transform
    for (int i = 0; i < arr->size; i += 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;

        float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
        float z1 = sqrt(-2.0f * log(u1)) * sin(2.0f * M_PI * u2);

        arr->data[i] = mean + std_dev * z0;
        if (i + 1 < arr->size) {
            arr->data[i + 1] = mean + std_dev * z1;
        }
    }

    return arr;
}

Arr* copy(Arr* inp) {
    Arr* result = malloc(sizeof(Arr));
    result->ndim = inp->ndim;
    result->size = inp->size;

    result->data = malloc(inp->size * sizeof(float));
    memcpy(result->data, inp->data, inp->size * sizeof(float));

    result->shape = malloc(inp->ndim * sizeof(int));
    memcpy(result->shape, inp->shape, inp->ndim * sizeof(int));

    result->strides = malloc(inp->ndim * sizeof(int));
    memcpy(result->strides, inp->strides, inp->ndim * sizeof(int));

    return result;
}

void conv2d(Arr* out, Arr* inp, Arr* kernel) {
    // Extract dimensions
    int batch_size = inp->shape[0];
    int C1 = inp->shape[1];
    int H = inp->shape[2];
    int W = inp->shape[3];
    int C2 = kernel->shape[0];
    int kernel_size = kernel->shape[2];  // Assuming square kernel

    // Calculate output dimensions
    int H_out = H - kernel_size + 1;
    int W_out = W - kernel_size + 1;

    // Perform convolution
    for (int b = 0; b < batch_size; b++) {
        for (int c2 = 0; c2 < C2; c2++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    float sum = 0.0f;
                    for (int c1 = 0; c1 < C1; c1++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int in_idx = b * inp->strides[0] + c1 * inp->strides[1] +
                                             (h + kh) * inp->strides[2] + (w + kw) * inp->strides[3];
                                int kern_idx = c2 * kernel->strides[0] + c1 * kernel->strides[1] +
                                               kh * kernel->strides[2] + kw * kernel->strides[3];
                                sum += inp->data[in_idx] * kernel->data[kern_idx];
                            }
                        }
                    }
                    int out_idx = b * out->strides[0] + c2 * out->strides[1] +
                                  h * out->strides[2] + w * out->strides[3];
                    out->data[out_idx] = sum;
                }
            }
        }
    }
}

void conv2d_backward(Arr* dinp, Arr* dkernel, Arr* dout, Arr* inp, Arr* kernel) {
    // Extract dimensions
    int batch_size = inp->shape[0];
    int C1 = inp->shape[1];
    int H = inp->shape[2];
    int W = inp->shape[3];
    int C2 = kernel->shape[0];
    int kernel_size = kernel->shape[2];  // Assuming square kernel

    int H_out = H - kernel_size + 1;
    int W_out = W - kernel_size + 1;

    // Compute gradients
    for (int b = 0; b < batch_size; b++) {
        for (int c2 = 0; c2 < C2; c2++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    int out_idx = b * dout->strides[0] + c2 * dout->strides[1] +
                                  h * dout->strides[2] + w * dout->strides[3];
                    float dout_val = dout->data[out_idx];

                    for (int c1 = 0; c1 < C1; c1++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                // Gradient w.r.t. input
                                int in_idx = b * inp->strides[0] + c1 * inp->strides[1] +
                                             (h + kh) * inp->strides[2] + (w + kw) * inp->strides[3];
                                int kern_idx = c2 * kernel->strides[0] + c1 * kernel->strides[1] +
                                               kh * kernel->strides[2] + kw * kernel->strides[3];
                                dinp->data[in_idx] += dout_val * kernel->data[kern_idx];

                                // Gradient w.r.t. kernel
                                dkernel->data[kern_idx] += dout_val * inp->data[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

void maxpool2d(Arr* out, Arr* inp, int kernel_size, int stride) {
    int B = inp->shape[0];
    int C = inp->shape[1];
    int H = inp->shape[2];
    int W = inp->shape[3];
    
    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;
    
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    float max_val = -INFINITY;
                    
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int h_inp = h * stride + kh;
                            int w_inp = w * stride + kw;
                            
                            int inp_idx = b * inp->strides[0] + c * inp->strides[1] +
                                          h_inp * inp->strides[2] + w_inp * inp->strides[3];
                            
                            float val = inp->data[inp_idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    
                    int out_idx = b * out->strides[0] + c * out->strides[1] +
                                  h * out->strides[2] + w * out->strides[3];
                    out->data[out_idx] = max_val;
                }
            }
        }
    }
}

void maxpool2d_backward(Arr* dinp, Arr* dout, Arr* inp, int kernel_size, int stride) {
    int B = inp->shape[0];
    int C = inp->shape[1];
    int H = inp->shape[2];
    int W = inp->shape[3];

    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;

    // Initialize dinp with zeros
    for (int i = 0; i < dinp->size; i++) {
        dinp->data[i] = 0.0f;
    }

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    int max_h = -1, max_w = -1;
                    float max_val = -INFINITY;

                    // Find the position of the maximum value in the input patch
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int h_inp = h * stride + kh;
                            int w_inp = w * stride + kw;

                            int inp_idx = b * inp->strides[0] + c * inp->strides[1] +
                                          h_inp * inp->strides[2] + w_inp * inp->strides[3];

                            float val = inp->data[inp_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_h = h_inp;
                                max_w = w_inp;
                            }
                        }
                    }

                    // Propagate the gradient to the input position where the maximum was found
                    if (max_h != -1 && max_w != -1) {
                        int out_idx = b * dout->strides[0] + c * dout->strides[1] +
                                      h * dout->strides[2] + w * dout->strides[3];
                        int inp_idx = b * dinp->strides[0] + c * dinp->strides[1] +
                                      max_h * dinp->strides[2] + max_w * dinp->strides[3];
                        dinp->data[inp_idx] += dout->data[out_idx];
                    }
                }
            }
        }
    }
}

/*
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15


 * */

int main() {
    Arr* a = create_arr((int[]){1,1,4,4}, 4);  
    Arr* kernel = create_arr((int[]){2,1,3,3}, 4);
    for (int i = 0; i < 16; i++) a->data[i] = i;
    for (int i = 0; i < 9; i++) kernel->data[i] = 2;
    for (int i = 9; i < 18; i++) kernel->data[i] = 1;
    Arr* out = create_arr((int[]){1,2,2,2},4);
    conv2d(out, a, kernel);
    for(int i = 0; i < 16; i++) printf("%f, ", a->data[i]); 
    printf("\n");
    for(int i = 0; i < 4; i++) printf("%d, ", a->strides[i]);
    printf("\n");
    for(int i = 0; i < 9; i++) printf("%f, ", kernel->data[i]); 
    printf("\n");
    for(int i = 0; i < 8; i++) printf("%f, ", out->data[i]); 
    printf("\n");
    Arr* poolout = create_arr((int[]){1,1,3,3},4);
    maxpool2d(poolout, a, 2, 1);
    printf("maxpool output:\n");
    for(int i = 0; i < 9; i++) printf("%f, ", poolout->data[i]); 
    free_arr(a);
    free_arr(kernel);
    free_arr(out);
    return 0;
}

