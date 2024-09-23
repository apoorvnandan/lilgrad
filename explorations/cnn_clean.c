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
void free_arr(Arr* a);

void matmul(Arr* c, Arr* a, Arr* b);
void matmul_backward(Arr* da, Arr* db, Arr* dc, Arr* a, Arr* b);
void relu(Arr* out, Arr* inp);
void relu_backward(Arr* dinp, Arr* dout, Arr* inp);
void logsoftmax(Arr* out, Arr* inp);
void logsoftmax_backward(Arr* dinp, Arr* dout, Arr* out);
void lossfn(Arr* loss, Arr* a, Arr* b);
void lossfn_backward(Arr* da, Arr* a, Arr* b);
void conv2d(Arr* out, Arr* inp, Arr* kernel);
void maxpool2d(Arr* out, Arr* inp, int kernel_size, int stride);
void flatten(Arr* out, Arr* inp);
void conv2d_backward(Arr* dinp, Arr* dkernel, Arr* dout, Arr* inp, Arr* kernel);
void maxpool2d_backward(Arr* dinp, Arr* dout, Arr* inp, int kernel_size, int stride);
void view(Arr* a, int* out_shape, int out_ndim);
void view_backward(Arr* da, int* in_shape, int in_ndim);


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
        out->data[i] = (inp->data[i] > 0) ? inp->data[i] : 0;
    }
}

void relu_backward(Arr* dinp, Arr* dout, Arr* inp) {
    for (int i = 0; i < inp->size; i++) {
        dinp->data[i] = (inp->data[i] > 0) ? dout->data[i] : 0;
    }
}


void logsoftmax(Arr* out, Arr* inp) {
    // inp and out are both (B,C)
    for (int b = 0; b < inp->shape[0]; b++) {
        float maxv = inp->data[b * inp->strides[0]];
        for (int c = 1; c < inp->shape[1]; c++) {
            int pos = b * inp->strides[0] + c * inp->strides[1];
            if (maxv < inp->data[pos]) {
                maxv = inp->data[pos];
            }
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
            int pos = b*out->shape[1] + c;
            dinp->data[pos] = dout->data[pos] - expf(out->data[pos]) * gradsum;
        }
    }
}


void lossfn(Arr* loss, Arr* a, Arr* b) {
    float s = 0.0f;
    for (int i = 0; i < a->size; i++) {
        s += a->data[i] * b->data[i] * (-1);    
    }
    loss->data[0] = s/a->size;
}

void lossfn_backward(Arr* da, Arr* a, Arr* b) {
    for (int i = 0; i < a->size; i++) {
        da->data[i] = b->data[i] * (-1);
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

void free_arr(Arr* a) {
    if (a) {
        free(a->data);
        free(a->shape);
        free(a->strides);
        free(a);
    }
}

float random_normal() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
}

// kaiming initialization
float kaiming_init(int fan_in) {
    float std_dev = sqrtf(2.0f / fan_in);
    return random_normal() * std_dev;
}

float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

float rand_range(float min, float max) {
    return min + rand_float() * (max - min);
}

// kaiming uniform initialization
float kaiming_uniform(int fan_in) {
    float gain = sqrtf(2.0f);  // for ReLU activation
    float std = gain / sqrtf(fan_in);
    float bound = sqrtf(3.0f) * std;
    return rand_range(-bound, bound);
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
                                int kern_idx = c2 * kernel->strides[0] + c1 * kernel->strides[1] +
                                               kh * kernel->strides[2] + kw * kernel->strides[3];
                                int in_idx = b * inp->strides[0] + c1 * inp->strides[1] +
                                             (h + kh) * inp->strides[2] + (w + kw) * inp->strides[3];
                                if (dinp != NULL) {
                                    dinp->data[in_idx] += dout_val * kernel->data[kern_idx];
                                }

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

void calculate_strides(int* shape, int ndim, int* strides) {
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

void view(Arr* a, int* out_shape, int out_ndim) {
    int has_neg_one = 0;
    int neg_one_index = -1;
    for (int i = 0; i < out_ndim; i++) {
        if (out_shape[i] == -1) {
            has_neg_one = 1;
            neg_one_index = i;
            break;
        }
    }

    // Compute the total size excluding -1 dimension
    int computed_size = 1;
    for (int i = 0; i < out_ndim; i++) {
        if (out_shape[i] != -1) {
            computed_size *= out_shape[i];
        }
    }

    if (has_neg_one) {
        // Calculate the size for the -1 dimension
        if (a->size % computed_size != 0) {
            // Error handling: if the new shape doesn't evenly divide the data size
            return;
        }
        out_shape[neg_one_index] = a->size / computed_size;
    } else {
        // If there's no -1, check if total size matches
        if (computed_size != a->size) {
            return; // Error, shapes don't match
        }
    }
    free(a->shape);
    free(a->strides);
    a->shape = malloc(out_ndim * sizeof(int));
    a->strides = malloc(out_ndim * sizeof(int));
    memcpy(a->shape, out_shape, out_ndim * sizeof(int));
    calculate_strides(a->shape, out_ndim, a->strides);
    a->ndim = out_ndim;
}

void view_backward(Arr* da, int* in_shape, int in_ndim) {
    view(da, in_shape, in_ndim);
}

void load_csv(Arr* input_arr, Arr* label_arr, char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(1);
    }

    char line[10000]; // Assuming no line will be longer than this
    char *token;
    
    for(int b = 0; b < 60000; b++) {
        if(fgets(line, sizeof(line), file) != NULL) {
            token = strtok(line, ",");
            for(int i = 0; i < 28*28 + 10; i++) {
                if (token == NULL) {
                    fprintf(stderr, "CSV format error: not enough columns\n");
                    fclose(file);
                    exit(1);
                }
                if(i < 28*28) {
                    input_arr->data[b * 28 * 28 + i] = atof(token);
                } else {
                    label_arr->data[b * 10 + (i - 28*28)] = atof(token);
                }
                token = strtok(NULL, ",");
            }
        } else {
            fprintf(stderr, "Not enough data for the specified batch size.\n");
            break;
        }
    }

    fclose(file);
}

void get_random_batch(Arr* batch_x, Arr* batch_y, Arr* x, Arr* y, int B) {
    static int seeded = 0;
    if (!seeded) {
        srand(0);
        seeded = 1;
    }
    if (B > x->shape[0] || B > y->shape[0]) {
        // Handle error: batch size too large
        return;
    }
    int *used_indices = (int *)calloc(x->shape[0], sizeof(int));
    for(int i = 0; i < B; i++) {
        int index;
        do {
            index = rand() % x->shape[0];  // Random index from 0 to 59999 for 60000 samples
        } while(used_indices[index]);  // Check if this index has been used
        used_indices[index] = 1;  // Mark this index as used
        for(int c = 0; c < x->shape[1]; c++) {
            for(int h = 0; h < x->shape[2]; h++) {
                for(int w = 0; w < x->shape[3]; w++) {
                    int x_index = (index * x->strides[0]) + (c * x->strides[1]) + (h * x->strides[2]) + (w * x->strides[3]);
                    int batch_x_index = (i * batch_x->strides[0]) + (c * batch_x->strides[1]) + (h * batch_x->strides[2]) + (w * batch_x->strides[3]);
                    batch_x->data[batch_x_index] = x->data[x_index];
                }
            }
        }
        for(int k = 0; k < 10; k++) {
            int y_index = index * y->strides[0] + k * y->strides[1];  // Assuming strides[1] would be 1 for class labels
            int batch_y_index = i * batch_y->strides[0] + k * batch_y->strides[1];
            batch_y->data[batch_y_index] = y->data[y_index];
        }
    }
    free(used_indices);
}

void train_batch(
        Arr* inp, Arr* labels, Arr* conv_kernel, Arr* linear_w, float lr,
        Arr* conv_out, Arr* relu_out, Arr* pool_out, Arr* linear_out, 
        Arr* logsoftmax_out, Arr* loss, Arr* logsoftmax_out_grad, 
        Arr* linear_out_grad, Arr* linear_w_grad, Arr* pool_out_grad,
        Arr* relu_out_grad, Arr* conv_out_grad, Arr* conv_kernel_grad
) {
    view(pool_out_grad, (int[]){-1, 10*12*12}, 4); // ensure correct shape
    view(pool_out, (int[]){-1, 10, 12, 12}, 4);
    conv2d(conv_out, inp, conv_kernel);
    relu(relu_out, conv_out);
    maxpool2d(pool_out, relu_out, 2, 2);
    view(pool_out, (int[]){-1,10*12*12}, 2);
    matmul(linear_out, pool_out, linear_w);
    logsoftmax(logsoftmax_out, linear_out);
    lossfn(loss, logsoftmax_out, labels);
    lossfn_backward(logsoftmax_out_grad, logsoftmax_out, labels);
    logsoftmax_backward(linear_out_grad, logsoftmax_out_grad, logsoftmax_out);
    matmul_backward(
        pool_out_grad, linear_w_grad, linear_out_grad, pool_out, linear_w
    );
    view_backward(pool_out_grad, (int[]){-1,10,12,12}, 4); 
    maxpool2d_backward(relu_out_grad, pool_out_grad, relu_out, 2, 2);
    relu_backward(conv_out_grad, relu_out_grad, conv_out);
    conv2d_backward(NULL, conv_kernel_grad, conv_out_grad, inp, conv_kernel);
    // update weights and zero grads
    for (int i = 0; i < conv_kernel->size; i++) {
        conv_kernel->data[i] = conv_kernel->data[i] - lr * conv_kernel_grad->data[i];
        conv_kernel_grad->data[i] = 0.0f;
    }
    for (int i = 0; i < linear_w->size; i++) {
        linear_w->data[i] = linear_w->data[i] - lr * linear_w_grad->data[i];
        linear_w_grad->data[i] = 0.0f;
    }
    memset(logsoftmax_out_grad->data, 0, logsoftmax_out_grad->size * sizeof(float));
    memset(linear_out_grad->data, 0, linear_out_grad->size * sizeof(float));
    memset(pool_out_grad->data, 0, pool_out_grad->size * sizeof(float));
    memset(relu_out_grad->data, 0, relu_out_grad->size * sizeof(float));
    memset(conv_out_grad->data, 0, conv_out_grad->size * sizeof(float));
}

void mnist() {
    srand(0);
    Arr* train_x = create_arr((int[]){60000,1,28,28}, 4);
    Arr* train_y = create_arr((int[]){60000,10}, 2);
    int B = 64;
    float lr = 0.001;
    Arr* conv_kernel = create_arr((int[]){10,1,5,5}, 4);
    for (int i = 0; i < conv_kernel->size; i++) conv_kernel->data[i] = kaiming_init(25);
    Arr* conv_kernel_grad = create_arr((int[]){10,1,5,5}, 4);
    Arr* linear_w = create_arr((int[]){10*12*12,10},2);
    for (int i = 0; i < linear_w->size; i++) linear_w->data[i] = kaiming_uniform(1440);
    Arr* linear_w_grad = create_arr((int[]){10*12*12,10},2);
    Arr* conv_out = create_arr((int[]){B, 10, 24, 24}, 4);
    Arr* conv_out_grad = create_arr((int[]){B, 10, 24, 24}, 4);
    Arr* relu_out = create_arr((int[]){B, 10, 24, 24}, 4);
    Arr* relu_out_grad = create_arr((int[]){B, 10, 24, 24}, 4);
    Arr* pool_out = create_arr((int[]){B, 10, 12, 12}, 4);
    Arr* pool_out_grad = create_arr((int[]){B, 10, 12, 12}, 4);
    Arr* linear_out = create_arr((int[]){B, 10}, 2);
    Arr* linear_out_grad = create_arr((int[]){B, 10}, 2);
    Arr* logsoftmax_out = create_arr((int[]){B, 10}, 2);
    Arr* logsoftmax_out_grad = create_arr((int[]){B, 10}, 2);
    Arr* loss = create_arr((int[]){1}, 1);
    load_csv(train_x, train_y, "mnist_train.csv");

    Arr* x = create_arr((int[]){B,1,28,28}, 4);
    Arr* y = create_arr((int[]){B,10},2);

    // zero grads
    for (int i = 0; i < conv_kernel->size; i++) {
        conv_kernel_grad->data[i] = 0.0f;
    }
    for (int i = 0; i < linear_w->size; i++) {
        linear_w_grad->data[i] = 0.0f;
    }
    memset(logsoftmax_out_grad->data, 0, logsoftmax_out_grad->size * sizeof(float));
    memset(linear_out_grad->data, 0, linear_out_grad->size * sizeof(float));
    memset(pool_out_grad->data, 0, pool_out_grad->size * sizeof(float));
    memset(relu_out_grad->data, 0, relu_out_grad->size * sizeof(float));
    memset(conv_out_grad->data, 0, conv_out_grad->size * sizeof(float));
    for (int i = 0; i < 5000; i++) {
        get_random_batch(x, y, train_x, train_y, B);    
        train_batch(
                x, y, conv_kernel, linear_w, lr, conv_out, relu_out, pool_out,
                linear_out, logsoftmax_out, loss, logsoftmax_out_grad, linear_out_grad, 
                linear_w_grad, pool_out_grad, relu_out_grad, conv_out_grad, conv_kernel_grad
        );
        if (i % 100 == 0) printf("batch: %d, loss: %f\n", i, loss->data[0]);
    }
    free_arr(train_x);
    free_arr(train_y);
    free_arr(conv_kernel);
    free_arr(conv_kernel_grad);
    free_arr(linear_w);
    free_arr(linear_w_grad);
    free_arr(conv_out);
    free_arr(conv_out_grad);
    free_arr(relu_out);
    free_arr(relu_out_grad);
    free_arr(pool_out);
    free_arr(pool_out_grad);
    free_arr(linear_out);
    free_arr(linear_out_grad);
    free_arr(logsoftmax_out);
    free_arr(logsoftmax_out_grad);
    free_arr(loss);
    free_arr(x); free_arr(y);
}


int main() {
    mnist();
    return 0;
}

