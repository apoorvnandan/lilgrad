#include <stdio.h>

void add(float* a, float *b, float *c, size_t n) {
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

