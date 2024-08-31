import ctypes
from time import time
import random
import numpy as np

lib = ctypes.CDLL('./ops.so')

lib.cpu_to_cuda.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int]
lib.cpu_to_cuda.restype = None

lib.cuda_to_cpu.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int]
lib.cuda_to_cpu.restype = None

lib.add_tensor_cuda.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.add_tensor_cuda.restype = None

N = 10000000
def generate_random_large_list():
    return [random.random() for _ in range(N)]

a = generate_random_large_list()
b = generate_random_large_list()

a_ctype = (ctypes.c_float * N)(*a)
a_ptr = ctypes.cast(a_ctype, ctypes.POINTER(ctypes.c_float))
a_ptr_ptr = ctypes.pointer(a_ptr)

b_ctype = (ctypes.c_float * N)(*b)
b_ptr = ctypes.cast(b_ctype, ctypes.POINTER(ctypes.c_float))
b_ptr_ptr = ctypes.pointer(b_ptr)

c_ctype = (ctypes.c_float * N)()
c_ptr = ctypes.cast(c_ctype, ctypes.POINTER(ctypes.c_float))
c_ptr_ptr = ctypes.pointer(c_ptr)

lib.cpu_to_cuda(a_ptr_ptr, N)
lib.cpu_to_cuda(b_ptr_ptr, N)
lib.cpu_to_cuda(c_ptr_ptr, N)

start_time = time()
lib.add_tensor_cuda(a_ptr, b_ptr, c_ptr, N)
print("CUDA time:", time() - start_time)

start_time = time()
c_numpy = (np.array(a) + np.array(b))
print("numpy time:", time() - start_time)

lib.cuda_to_cpu(c_ptr_ptr, N)
c = list(c_ctype)
c_ref = [a[i] + b[i] for i in range(N)] # to verify the results
print("maximum difference:", max(abs(c[i] - c_ref[i]) for i in range(N)))
