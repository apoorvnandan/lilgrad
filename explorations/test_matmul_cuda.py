import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
from time import time
from ctypes import c_float, c_int, POINTER, Structure

# Load the CUDA library
cuda_lib = ctypes.CDLL('./matmul.so')
class Arr(Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("cuda_data", POINTER(c_float)),
        ("shape", POINTER(c_int)),
        ("strides", POINTER(c_int)),
        ("ndim", c_int),
        ("size", c_int)
    ]

# Define cpu_to_cuda function
cuda_lib.cpu_to_cuda.argtypes = [POINTER(Arr)]
cuda_lib.cpu_to_cuda.restype = None

# Define cuda_to_cpu function
cuda_lib.cuda_to_cpu.argtypes = [POINTER(Arr)]
cuda_lib.cuda_to_cpu.restype = None

# Define matmul_arr function
cuda_lib.matmul_arr.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr)]
cuda_lib.matmul_arr.restype = ctypes.c_int

def numpy_to_ctype(np_array):
    data = list(np_array.flatten())
    data_array = (ctypes.c_float * np_array.size)(*data)
    return ctypes.cast(data_array, ctypes.POINTER(ctypes.c_float))

def numpy_to_arr(np_array):
    arr = Arr()
    data = list(np_array.flatten())
    data_array = (ctypes.c_float * np_array.size)(*data)
    arr.data = ctypes.cast(data_array, ctypes.POINTER(ctypes.c_float))
    # Create a C array for shape
    shape_array = (ctypes.c_int * np_array.ndim)(*list(np_array.shape))
    arr.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int))
    # Create a C array for strides
    strides_array = (ctypes.c_int * np_array.ndim)(*[stride // np_array.itemsize for stride in np_array.strides])
    arr.strides = ctypes.cast(strides_array, ctypes.POINTER(ctypes.c_int))
    arr.ndim = np_array.ndim
    arr.size = np_array.size
    return arr

def run_example():
    N = 1000

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.empty((N, N), dtype=np.float32)

    A_arr = numpy_to_arr(A)
    B_arr = numpy_to_arr(B)
    C_arr = numpy_to_arr(C)

    print(A_arr.data[5])

    cuda_lib.cpu_to_cuda(ctypes.byref(A_arr))
    cuda_lib.cpu_to_cuda(ctypes.byref(B_arr))
    cuda_lib.cpu_to_cuda(ctypes.byref(C_arr))

    start_time = time()
    error_code = cuda_lib.matmul_arr(ctypes.byref(C_arr), ctypes.byref(A_arr), ctypes.byref(B_arr))
    if error_code != 0:
        raise RuntimeError(f"CUDA error occurred. Error code: {error_code}")
    print("CUDA time:", time() - start_time)

    cuda_lib.cuda_to_cpu(ctypes.byref(A_arr))
    cuda_lib.cuda_to_cpu(ctypes.byref(B_arr))
    cuda_lib.cuda_to_cpu(ctypes.byref(C_arr))

    print(A_arr.data[5])

    C_list = [C_arr.data[i] for i in range(C_arr.size)]
    C_cuda = np.array(C_list).reshape(N, N)

    # NumPy matrix multiplication
    start_time = time()
    C_numpy = np.dot(A, B)
    print("NumPy time:", time() - start_time)

    # Check if results are close
    c_np_list = list(C_numpy.flatten())
    for i in range(C_arr.size):
        if abs(c_np_list[i] - C_arr.data[i]) > 0.001:
            print('breaking at', i)
            break
    print('DIFF:', np.sum(np.abs(C_cuda - C_numpy)))
    # print(C_cuda, C_numpy)
    print("Results match:", np.allclose(C_cuda, C_numpy, rtol=1e-4))

run_example()
