def generate_large_random_list():
    list_data = []
    for i in range(1000000):
        list_data.append(random.random())
    return list_data
     

import ctypes
from time import time
import random
import numpy as np # to benchmark against

lib = ctypes.CDLL('./ops.so')
lib.add.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.add.restype = None

list_data = generate_large_random_list()   
c_compatible_arr1 = (ctypes.c_float * len(list_data))(*list_data)
c_compatible_arr2 = (ctypes.c_float * len(list_data))(*list_data)

c_compatible_result = (ctypes.c_float * len(list_data))()
stime = time()
lib.add(c_compatible_arr1, c_compatible_arr2, c_compatible_result, len(list_data))
print(time() - stime, 'seconds using c code')

numpy_arr1 = np.array(list_data)
numpy_arr2 = np.array(list_data)
stime = time()
numpy_result = numpy_arr1 + numpy_arr2
print(time() - stime, 'seconds using numpy')

stime = time()
pure_python_result = [list_data[i] + list_data[i] for i in range(len(list_data))]
print(time() - stime, 'seconds using pure python')
