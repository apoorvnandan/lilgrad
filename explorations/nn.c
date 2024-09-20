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

void matmul(Arr* c, Arr* a, Arr* b);
void matmul_backward(float* da, float* db, float* dc, Arr* a, Arr* b);
void relu(Arr* out, Arr* inp);
void relu_backward(float* dinp, float* dout, Arr* out);
void logsoftmax(Arr* out, Arr* inp);
void logsoftmax_backward(float* dinp, float* dout, Arr* out);
void lossfn(Arr* loss, Arr* a, Arr* b);
void lossfn_backward(float* da, Arr* a, Arr* b);

void train_batch(
        Arr* inp, 
        Arr* labels,
        Arr* w1,
        Arr* w2,
        float lr,
        Arr* w1_out, Arr* relu_out, Arr* w2_out,
        Arr* logsoftmax_out, Arr* loss,
        float* logsoftmax_out_grad,
        float* w2_out_grad, float* w2_grad,
        float* relu_out_grad, float* w1_out_grad,
        float* w1_grad
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
        w1->data[i] = w1->data[i] - lr * w1_grad[i];
        w1_grad[i] = 0.0f;
    }
    for (int i = 0; i < w2->size; i++) {
        w2->data[i] = w2->data[i] - lr * w2_grad[i];
        w2_grad[i] = 0.0f;
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

void matmul_backward(float* da, float* db, float* dc, Arr* a, Arr* b) {
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
                    tmp += dc[i * R + k] * b->data[pos_b];
                }
                int pos_da = i * Q + j;
                da[pos_da] = tmp;
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
                    tmp += dc[k * R + j] * a->data[pos_a];
                }
                int pos_db = i * R + j;
                db[pos_db] = tmp;
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

void relu_backward(float* dinp, float* dout, Arr* out) {
    for (int i = 0; i < out->size; i++) {
        if (out->data[i] > 0) {
            dinp[i] = dout[i];
        } else {
            dinp[i] = 0.0f;
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

void logsoftmax_backward(float* dinp, float* dout, Arr* out) {
    // dout and dinp are both (B,C)
    for (int b = 0; b < out->shape[0]; b++) {
        float gradsum = 0.0f;
        for (int c = 0; c < out->shape[1]; c++) {
            gradsum += dout[b * out->shape[1] + c];
        }
        for (int c = 0; c < out->shape[1]; c++) {
            int pos = b*out->shape[1]+c;
            dinp[pos] = dout[pos] - expf(out->data[pos]) * gradsum;
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

void lossfn_backward(float* da, Arr* a, Arr* b) {
    for (int i = 0; i < a->size; i++) {
        da[i] = b->data[i];
    }
}
