import numpy as np
import ctypes

lib = ctypes.CDLL('./ops.so')
lib.add_broadcasted.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.c_int,
]
lib.add_broadcasted.restype = ctypes.c_int

a_np = np.random.rand(4,3,3,1)
b_np = np.random.rand(8)
c_np = a_np + b_np

a_list = list(a_np.flatten())
b_list = list(b_np.flatten())
c_list = list(c_np.flatten())

a = (ctypes.c_float * len(a_list))(*a_list)
b = (ctypes.c_float * len(b_list))(*b_list)
shape_a = (ctypes.c_int * 4)(*[4,3,3,1])
strides_a = (ctypes.c_int * 4)(*[9,3,1,1])
shape_b = (ctypes.c_int * 1)(*[8])
strides_b = (ctypes.c_int * 1)(*[1])
shape_c = (ctypes.c_int * 4)(*[4,3,3,8])
strides_c = (ctypes.c_int * 4)(*[72,24,8,1])
c = (ctypes.c_float * 288)()
err = lib.add_broadcasted(
    a, shape_a, strides_a, 4, b, shape_b, strides_b, 1, c, shape_c, strides_c, 4
)
print(err)
if err == 0:
    calculated_c = list(c)
    print(sum([abs(c_list[i] - calculated_c[i]) for i in range(4*3*3*8)]))
    


