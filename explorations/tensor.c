#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define MAX_PREVS 3
#define MAX_ARGS 5
#define MAX_PARAM_TENSORS 10
// op codes
#define MATMUL 0
#define MEAN 1
#define MUL 2

typedef struct {
    float* values;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;

typedef union {
    int ival;
    float fval;
} Arg;

typedef struct Tensor {
    Arr* data;
    Arr* grad;
    int op; // op used to create this tensor
    struct Tensor* prevs[MAX_PREVS]; // tensors that were processed by the op
    int num_prevs;
    Arg args[MAX_ARGS]; // additional args for the op (e.g. axis, stride etc.)
} Tensor;

Arr* create_arr(float* data, int* shape, int ndim);
Arr* create_arr_zeros(int *shape, int ndim);
void free_arr(Arr* a);
Tensor* create_tensor(float* data, int* shape, int ndim);
void free_tensor(Tensor* t);
void backward(Tensor* t);

Tensor* mul(Tensor* a, Tensor* b);
void mul_backward(Tensor* out);
Tensor* mean(Tensor* a);
void mean_backward(Tensor* out);
void print_tensor(Tensor* t);

void print_tensor(Tensor* t) {
    printf("Tensor(\n");
    printf("\tdata: ");
    for (int i = 0; i < t->data->size; i++) printf("%f,", t->data->values[i]);
    printf("\n\tshape: ");
    for (int i = 0; i < t->data->ndim; i++) printf("%d,", t->data->shape[i]);
    printf("\n\tgrad: ");
    for (int i = 0; i < t->data->size; i++) printf("%f,", t->grad->values[i]);
    printf("\n)\n");
}

Arr* create_arr(float* data, int* shape, int ndim) {
    Arr* arr = create_arr_zeros(shape, ndim);
    memcpy(arr->values, data, arr->size * sizeof(float));
    return arr;
}

Arr* create_arr_zeros(int* shape, int ndim) {
    Arr* arr = (Arr*) malloc(sizeof(Arr));
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

    arr->values = (float*)calloc(arr->size, sizeof(float));
    if (!arr->values) {
        free(arr->strides);
        free(arr->shape);
        free(arr);
        return NULL;
    }

    return arr;
}

Tensor* create_tensor(float* data, int* shape, int ndim) {
    Arr* d = create_arr(data, shape, ndim);
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = d;
    t->grad = create_arr_zeros(shape, ndim);
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

void free_arr(Arr* a) {
    if (a == NULL) return;
    if (a->values != NULL) {
        free(a->values);
    }
    if (a->shape != NULL) {
        free(a->shape);
    }
    if (a->strides != NULL) {
        free(a->strides);
    }
    free(a);
}

void free_tensor(Tensor* t) {
    if (t == NULL) return;
    if (t->data != NULL) free_arr(t->data);
    if (t->grad != NULL) free_arr(t->grad);
    free(t);
}


void backward(Tensor* t) {
    // assumes that the grad of `t` has been computed
    // and computes the grad for tensors in `t->prevs`
    // then calls the backward function on prev tensors
    if (t->op == MUL) {
        mul_backward(t);
    } else if (t->op == MEAN) {
        mean_backward(t);
    }
    for (int i = 0; i < t->num_prevs; i++) {
        backward(t->prevs[i]);
    }
}

Tensor* mul(Tensor* a, Tensor* b) {
    float* d = (float*) malloc(a->data->size * sizeof(float));
    for (int i = 0; i < a->data->size; i++) {
        d[i] = a->data->values[i] * b->data->values[i];
    }
    Tensor* t = create_tensor(d, a->data->shape, a->data->ndim);
    t->op = MUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

void mul_backward(Tensor* out) {
    for (int i = 0; i < out->data->size; i++) {
        out->prevs[0]->grad->values[i] += out->grad->values[i] * out->prevs[1]->data->values[i];
        out->prevs[1]->grad->values[i] += out->grad->values[i] * out->prevs[0]->data->values[i];
    }
}

Tensor* mean(Tensor* t) {
    float* d = (float*) malloc(sizeof(float));
    float s = 0.0f;
    for(int i = 0; i < t->data->size; i++) s += t->data->values[i];
    d[0] = s/t->data->size;
    Tensor* m = create_tensor(d, (int[]){1}, 1);
    m->op = MEAN;
    m->num_prevs = 1;
    m->prevs[0] = t;
    return m;
}

void mean_backward(Tensor* out) {
    for (int i = 0; i < out->prevs[0]->grad->size; i++) {
        out->prevs[0]->grad->values[i] += out->grad->values[0] / out->prevs[0]->data->size;
    }
}

int main() {
    Tensor* a = create_tensor((float[]){1.0, 2.0, 3.0, 4.0}, (int []){2,2}, 2);
    Tensor* b = create_tensor((float[]){2.0, 2.0, 2.0, 2.0}, (int []){2,2}, 2);
    Tensor* c = mul(a, b);
    Tensor* d = mean(c);
    d->grad->values[0] = 1.0f; // set d->grad to one
    backward(d);
    print_tensor(d);
    print_tensor(a); // observe grad values
    print_tensor(b); // observe grad values

    // free memory - can be replaced by gc later
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    free_tensor(d);
    return 0;
}
