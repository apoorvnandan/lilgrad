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

void check_bool(float* a, float* result, size_t totalsize, float value, char op) {
    for (size_t i = 0; i < totalsize; i++) {
        if (op == '<') result[i] = (a[i] < value) ? 1.0f : 0.0f;
        if (op == '=') result[i] = (a[i] == value) ? 1.0f : 0.0f;
        if (op == '>') result[i] = (a[i] > value) ? 1.0f : 0.0f;
    } 
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
    int* indices_a = (int*) calloc(ndim_a, sizeof(int));
    int* indices_b = (int*) calloc(ndim_b, sizeof(int));
    for (int i = 0; i < size_result; i++) {
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
    }
    free(indices_a);
    free(indices_b);
    free(indices);
    return 0;
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

void sum_reduce(float* a, int* shape, int* strides, int ndim, float* result, int* shape_result, int* strides_result, int dim) {
    int* indices = calloc(ndim, sizeof(int));
    int result_size = 1;
    for (int i = 0; i < ndim; i++) {
        result_size *= shape_result[i];
    }

    for (int i = 0; i < result_size; i++) {
        float value = 0.0f;
        int x = indices[dim];
        for (int j = 0; j < shape[dim]; j++) {
            indices[dim] = j;
            int pos_a = get_flat_index(indices, strides, ndim);
            value += a[pos_a];
        }
        indices[dim] = x;
        int pos_result = get_flat_index(indices, strides_result, ndim);
        result[pos_result] = value;

        // Increment indices
        for (int j = ndim - 1; j >= 0; j--) {
            if (j == dim) continue;  // Skip the reduction dimension
            indices[j]++;
            if (indices[j] < shape_result[j]) {
                break;
            }
            indices[j] = 0;
        }
    }

    free(indices);
}


void max_reduce(float* a, int* shape, int* strides, int ndim, float* result, int* shape_result, int* strides_result, int dim) {
    int* indices = calloc(ndim, sizeof(int));
    int result_size = 1;
    for (int i = 0; i < ndim; i++) {
        result_size *= shape_result[i];
    }

    for (int i = 0; i < result_size; i++) {
        float value = -INFINITY;
        int x = indices[dim];
        for (int j = 0; j < shape[dim]; j++) {
            indices[dim] = j;
            int pos_a = get_flat_index(indices, strides, ndim);
            if (a[pos_a] > value) {
                value = a[pos_a];
            }
        }
        indices[dim] = x;
        int pos_result = get_flat_index(indices, strides_result, ndim);
        result[pos_result] = value;

        // Increment indices
        for (int j = ndim - 1; j >= 0; j--) {
            if (j == dim) continue;  // Skip the reduction dimension
            indices[j]++;
            if (indices[j] < shape_result[j]) {
                break;
            }
            indices[j] = 0;
        }
    }

    free(indices);
}

void transpose(float* a, int* shape, int* strides, int ndim, size_t totalsize, float* result, int* shape_result, int* strides_result) {
    int* indices = (int*) calloc(ndim, sizeof(int));
    int* new_indices = (int*) calloc(ndim, sizeof(int));
    for (size_t i = 0; i < totalsize; i++) {
        int old_pos = get_flat_index(indices, strides, ndim);
        
        // Calculate new index based on reversed dimensions
        for (int j = ndim-1; j >= 0; j--) {
            new_indices[ndim-1-j] = indices[j];
        }

        int result_pos = get_flat_index(new_indices, strides_result, ndim);
        
        result[result_pos] = a[old_pos];
        
        // Increment indices
        for (int j = ndim - 1; j >= 0; j--) {
            indices[j]++;
            if (indices[j] < shape[j]) {
                break;
            }
            indices[j] = 0;
        }
    }
    free(new_indices);
    free(indices);
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

int matmul(
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
    // is matmul possible
    if (ndim_b == 1) {
        if (shape_a[ndim_a - 1] != shape_b[0]) {
            return 1;
        }
    } else {
        if (shape_a[ndim_a - 1] != shape_b[ndim_b - 2]) {
            return 1;
        }
    }
    for (int dim_from_right = 2; dim_from_right < ndim_result; dim_from_right++) {
        if (dim_from_right < ndim_a && dim_from_right < ndim_b) {
            int dim_a = ndim_a - 1 - dim_from_right;
            int dim_b = ndim_b - 1 - dim_from_right;
            if (shape_a[dim_a] != shape_b[dim_b]) {
                return 1;
            }
        }
    } 
    // handle common cases with optimised code
    if (ndim_a == 2 && ndim_b == 2) {
        // (P,Q) x (Q,R)
        int P = shape_a[0]; 
        int Q = shape_a[1];
        int R = shape_b[1];
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < R; j++) {
                float val = 0.0f;
                for (int k = 0; k < Q; k++) {
                    int pos_a = i * strides_a[0] + k * strides_a[1];
                    int pos_b = k * strides_b[0] + j * strides_b[1];
                    val += a[pos_a] * b[pos_b];
                }
                int pos_result = i * strides_result[0] + j * strides_result[1];
                result[pos_result] = val;
            }
        }
        return 0;
    } 
    // calculating the result values
    int* indices_a = (int*) calloc(ndim_a, sizeof(int));
    int* indices_b = (int*) calloc(ndim_b, sizeof(int));
    int* indices = (int *) calloc(ndim_result, sizeof(int));
    int totalsize = 1;
    for (int i = 0; i < ndim_result; i++) {
        totalsize *= shape_result[i];
    }
    for (int i = 0; i < totalsize; i++) {
        float value = 0.0f;
        for (int j = 0; j < shape_a[ndim_a - 1]; j++) {
            indices_a[ndim_a - 1] = j;
            if (ndim_b == 1) {indices_b[0] = j;}
            else {indices_b[ndim_b - 2] = j;}
            for (int dim_from_right = 0; dim_from_right < ndim_result; dim_from_right++) {
                int dim_from_left = ndim_result - 1 - dim_from_right;
                if (dim_from_right == 0) {
                    if (ndim_b == 1) {
                        indices_b[0] = j;
                    } else {
                        indices_b[ndim_b - 1] = indices[dim_from_left];
                    }
                    indices_a[ndim_a - 1] = j;
                    continue;
                }
                if (dim_from_right == 1) {
                    if (ndim_b != 1) {
                        indices_b[ndim_b - 2] = j;
                    }
                    indices_a[ndim_a - 2] = indices[dim_from_left];
                    continue;
                }
                if (dim_from_right < ndim_a) {
                    int dim_a = ndim_a - 1 - dim_from_right;
                    indices_a[dim_a] = indices[dim_from_left];
                }
                if (dim_from_right < ndim_b) {
                    int dim_b = ndim_b - 1 - dim_from_right;
                    indices_b[dim_b] = indices[dim_from_left];
                }
            }
            int pos_a = get_flat_index(indices_a, strides_a, ndim_a);
            int pos_b = get_flat_index(indices_b, strides_b, ndim_b);
            value += a[pos_a] * b[pos_b];
        }
        int pos_result = get_flat_index(indices, strides_result, ndim_result);
        result[pos_result] = value;
        // increment indices
        for (int j = ndim_result - 1; j >= 0; j--) {
            indices[j]++;
            if (indices[j] < shape_result[j]) {
                break;
            }
            indices[j] = 0;
        }
    }
    free(indices_a);
    free(indices_b);
    free(indices);
    return 0;
}
