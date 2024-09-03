#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int get_flat_index(int* indices, int* strides, int ndim) {
    int flat_index = 0;
    for (int i = 0; i < ndim; i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
}

int op_broadcasted(
        char op,
        float* a,
        int* shape_a,
        int* strides_a,
        int ndim_a,
        float* b,
        int* shape_b,
        int* strides_b,
        int ndim_b,
        float* result,
        int* shape_result,
        int* strides_result,
        int ndim_result
) {
    // is broadcasting possible
    for (int dim = 0; dim < ndim_result; dim++) {
        if (dim < ndim_a && dim < ndim_b) {
            int dim_a = ndim_a - 1 - dim;
            int dim_b = ndim_b - 1 - dim;
            if (shape_a[dim_a] != shape_b[dim_b]) {
                if (shape_a[dim_a] != 1 && shape_b[dim_b] != 1) {
                    printf("shapes cannot be broadcasted ndim_a %d ndim_b %d shape_a[ndim_a] %d shape_b[ndim_b] %d", ndim_a, ndim_b, shape_a[ndim_a], shape_b[ndim_b]);
                    return 1;
                }
            }
        }
    }
    
    // calculate result values
    int size_result = 1;
    for (int i = 0; i < ndim_result; i++) {
        size_result *= shape_result[i];
    }
    int* indices = (int*) calloc(ndim_result, sizeof(int));
    for (int i = 0; i < size_result; i++) {
        int* indices_a = (int*) calloc(ndim_a, sizeof(int));
        int* indices_b = (int*) calloc(ndim_b, sizeof(int));
        for (int d = ndim_result - 1; d >= 0; d--) {
            int dim_from_right = ndim_result - 1 - d;
            if (dim_from_right < ndim_a) {
                int dim_a = ndim_a - 1 - dim_from_right;
                if (indices[d] >= shape_a[dim_a]) {
                    indices_a[dim_a] = 0; // because shape_a[dim_a] is 1
                } else {
                    indices_a[dim_a] = indices[d];
                }
            }
            if (dim_from_right < ndim_b) {
                int dim_b = ndim_b - 1 - dim_from_right;
                if (indices[d] >= shape_b[dim_b]) {
                    indices_b[dim_b] = 0; // because shape_b[dim_b] is 1
                } else {
                    indices_b[dim_b] = indices[d];
                }
            }
        }
        int idx_a = get_flat_index(indices_a, strides_a, ndim_a);
        int idx_b = get_flat_index(indices_b, strides_b, ndim_b);
        int idx_result = get_flat_index(indices, strides_result, ndim_result);
        if (op == '+') {result[idx_result] = a[idx_a] + b[idx_b];}
        if (op == '-') {result[idx_result] = a[idx_a] - b[idx_b];}
        if (op == '*') {result[idx_result] = a[idx_a] * b[idx_b];}

        // increment indices
        for (int j = ndim_result - 1; j >= 0; j--) {
            indices[j]++;
            if (indices[j] < shape_result[j]) {
                break;
            }
            indices[j] = 0;
        }
        free(indices_a);
        free(indices_b);
    }
    free(indices);
    return 0;
}

void add(float* a, float *b, float *c, size_t na, size_t nb) {
    if (na == nb) {
        for(size_t i = 0; i < na; i++) {
            c[i] = a[i] + b[i];
        }
    } else if (nb == 1) {
        for(size_t i = 0; i < na; i++) {
            c[i] = a[i] + b[0];
        }
    }
}

void sub(float* a, float *b, float *c, size_t na, size_t nb) {
    if (na == nb) {
        for(size_t i = 0; i < na; i++) {
            c[i] = a[i] - b[i];
        }
    } else if (nb == 1) {
        for(size_t i = 0; i < na; i++) {
            c[i] = a[i] - b[0];
        }
    }
}

void mul(float *a, float *b, float *c, size_t na, size_t nb) {
    if (na == nb) {
        for(size_t i = 0; i < na; i++) {
            c[i] = a[i] * b[i];
        }
    } else if (nb == 1) {
        for(size_t i = 0; i < na; i++) {
            c[i] = a[i] * b[0];
        }
    }
}

void maximum(float* a, float* result, float value, size_t n) {
    for(size_t i = 0; i < n; i++) {
        if (a[i] > value) {
            result[i] = a[i];
        } else {
            result[i] = value;
        }
    } 
}

void mean(float *a, float* result, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        s += a[i];
    }
    s = s/n;
    *result = s;
}

void sum(float *a, float* result, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        s += a[i];
    }
    *result = s;
}

void expfloat(float *a, float *result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = expf(a[i]);
    }
}

void logfloat(float *a, float* result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = logf(a[i]);
    }
}

void ones(float *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0f;
    }
}

void zeros(float *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = 0.0f;
    }
}

void matmul(
        float* a, 
        float* b, 
        float* result, 
        int* shape_a,
        int ndim_a,
        int* shape_b,
        int ndim_b
) {
    // will support only (t x c) x (c x d) and (b x t x c) x (c x d) for now
    // what about (b x t x h x w) x (w x d)
    // broadcasted
    
}
