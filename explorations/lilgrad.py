import ctypes
from time import time
import random
import math

lib = ctypes.CDLL('./ops.so')
lib.op_broadcasted.argtypes = [
        ctypes.c_char,
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
lib.op_broadcasted.restype = ctypes.c_int

lib.add.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t
]
lib.add.restype = None

lib.sub.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.c_size_t
]
lib.sub.restype = None

lib.mul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t
]
lib.sub.restype = None

lib.logfloat.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.logfloat.restype = None

lib.expfloat.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.expfloat.restype = None

lib.mean.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_size_t
]
lib.mean.restype = None

lib.sum.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_size_t
]
lib.sum.restype = None

lib.maximum.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_size_t
]
lib.maximum.restype = None

lib.zeros.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.zeros.restype = None

lib.ones.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.ones.restype = None

def create_c_compat_array(data, dtype='float'):
    if isinstance(data, list):
        if dtype == 'float':
            return (ctypes.c_float * len(data))(*data)
        elif dtype == 'int':
            return (ctypes.c_int * len(data))(*data)
    else: # todo - add check for ctype array
        return data

def create_strides_from_shape(shape, itemsize=1):
    " shape is expected to be a list here "
    strides = [itemsize]
    for s in shape[1:][::-1]:
        strides.append(strides[-1] * s)
    return list(reversed(strides))

class Array:
    def __init__(self, data, shape, dtype='float'):
        self.data = create_c_compat_array(data, dtype)
        self.shape = create_c_compat_array(shape, 'int')
        self.strides = create_c_compat_array(create_strides_from_shape(shape), 'int')
        self.ndim = len(shape)
        self.size = len(data)
        self.device = 'cpu'

    @staticmethod
    def zeros_like(x):
        return Array([0] * x.size, x.shape)

    @staticmethod
    def ones_like(x):
        return Array([1] * x.size, x.shape)

    @staticmethod
    def randn(shape, mean=0, std_dev=1):
        def normal(mean, std_dev):
            u1 = random.random()
            u2 = random.random()
            r = math.sqrt(-2 * math.log(u1))
            theta = 2 * math.pi * u2
            z0 = r * math.cos(theta)
            return z0 * std_dev + mean
        totalsize = 1
        for s in shape:
            totalsize *= s
        d = [normal(mean, std_dev) for i in range(totalsize)]
        return Array(d, shape)

    def elementwise_op(self, other, op):
        ndim_result = max(self.ndim, other.ndim)
        shape_result = []
        for dim in range(ndim_result):
            dim_a = self.ndim - 1 - dim
            dim_b = other.ndim - 1 - dim
            shape_a_at_dim = 1
            if dim_a >= 0:
                shape_a_at_dim = self.shape[dim_a]
            shape_b_at_dim = 1
            if dim_b >= 0:
                shape_b_at_dim = other.shape[dim_b]
            shape_result.insert(0, max(shape_a_at_dim, shape_b_at_dim))
        strides_result = create_strides_from_shape(shape_result)
        shape = create_c_compat_array(shape_result, 'int')
        strides = create_c_compat_array(strides_result, 'int')
        size_result = 1
        for s in shape_result:
            size_result *= s
        d = (ctypes.c_float * size_result)()
        err = lib.op_broadcasted(
                op, self.data, self.shape, self.strides, self.ndim,
                other.data, other.shape, other.strides, other.ndim,
                d, shape, strides, ndim_result
        )
        if (err == 1):
            assert False, "shapes cannot be broadcasted during addition"
        return Array(d, shape)

    def __add__(self, other):
        return self.elementwise_op(other, b'+')

    def __sub__(self, other):
        return self.elementwise_op(other, b'-')

    def __mul__(self, other):
        return self.elementwise_op(other, b'*')

    def __str__(self):
        return str(list(self.data))

    def list(self):
        return list(self.data)
    
    def exp(self):
        d = (ctypes.c_float * self.size)()
        lib.expfloat(self.data, d, self.size)
        return Array(d, self.shape)

    def log(self):
        d = (ctypes.c_float * self.size)()
        lib.logfloat(self.data, d, self.size)
        return Array(d, self.shape)

    @staticmethod
    def sum(x):
        d = (ctypes.c_float)()
        lib.sum(x.data, d, self.size)
        return Array(d, [1])
    
    @staticmethod
    def mean(x):
        d = (ctypes.c_float)()
        lib.mean(x.data, d, self.size)
        return Array(d, [1])

    def maximum(x, value):
        d = (ctypes.c_float * x.size)()
        lib.maximum(x, d, value, x.size)
        return Array(d, x.shape)

def logsoftmax(x):
    pass


x = Array.randn([8,4,1])
y = Array.randn([4])
z = x + y
z1 = x - y
z2 = x * y

import numpy as np

a = np.array(x.list()).reshape([8,4,1])
b = np.array(y.list()).reshape([4])
c = a + b
c1 = a - b
c2 = a * b

def compare(nparr, arr):
    p = list(nparr.flatten())
    q = arr.list()
    print(sum([abs(p[i] - q[i]) for i in range(len(p))]))

compare(c, z)
compare(c1, z1)
compare(c2, z2)

