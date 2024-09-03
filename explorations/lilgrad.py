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

lib.matmul.argtypes = [
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
lib.matmul.restype = ctypes.c_int


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

def calculate_matmul_shape(shape_a_ctype, shape_b_ctype):
    shape_a = list(shape_a_ctype)
    shape_b = list(shape_b_ctype)
    # Ensure shape_a has at least two dimensions
    if len(shape_a) < 2:
        raise ValueError("Matrix A must be at least 2-dimensional for matrix multiplication.")
    
    # If shape_b has only one dimension, treat it as a column vector for multiplication
    if len(shape_b) == 1:
        if shape_b[0] != shape_a[-1]:
            raise ValueError(f"Cannot multiply: the number of elements in B ({shape_b[0]}) does not match the last dimension of A ({shape_a[-1]}).")
        # The result will be the shape of A without the last dimension
        return shape_a[:-1]
    
    # Check if standard matrix multiplication is possible
    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f"Cannot multiply the matrices. Dimension mismatch: {shape_a[-1]} vs {shape_b[-2]}")

    # Construct the result shape
    result_shape = shape_a[:-2] + [shape_a[-2]] + (list(shape_b[-1:]) if len(shape_b) > 1 else [])
    
    # Handle broadcasting for extra dimensions
    broadcast_shape_a = shape_a[:-2]
    broadcast_shape_b = shape_b[:-2] if len(shape_b) > 1 else [1]  # Treat B as having a leading 1 for broadcasting if it's 1D or 2D
    
    max_dim = max(len(broadcast_shape_a), len(broadcast_shape_b))
    broadcast_shape_a = [1] * (max_dim - len(broadcast_shape_a)) + list(broadcast_shape_a)
    broadcast_shape_b = [1] * (max_dim - len(broadcast_shape_b)) + list(broadcast_shape_b)

    for i in range(max_dim):
        if broadcast_shape_a[i] != 1 and broadcast_shape_b[i] != 1 and broadcast_shape_a[i] != broadcast_shape_b[i]:
            raise ValueError(f"Broadcasting error: non-matmul dimensions do not match or are not 1 at dimension {i}")

    return result_shape

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

    def __add__(self, other): return self.elementwise_op(other, b'+')

    def __sub__(self, other): return self.elementwise_op(other, b'-')

    def __mul__(self, other): return self.elementwise_op(other, b'*')

    @staticmethod
    def matmul(a, b):
        shape_result = calculate_matmul_shape(a.shape, b.shape) 
        strides_result = create_strides_from_shape(shape_result)
        shape = create_c_compat_array(shape_result, 'int')
        strides = create_c_compat_array(strides_result, 'int')
        size_result = 1
        for s in shape_result:
            size_result *= s
        d = (ctypes.c_float * size_result)()
        err = lib.matmul(
                a.data, a.shape, a.strides, a.ndim,
                b.data, b.shape, b.strides, b.ndim,
                d, shape, strides, len(shape_result)
        )
        if err == 1:
            assert False
        return Array(d, shape)

    def __str__(self): return str(list(self.data))

    def list(self): return list(self.data)
    
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
        lib.maximum(x.data, d, value, x.size)
        return Array(d, x.shape)
    




class FeedForwardNet:
    def __init__(self):
        self.w1 = Array.randn([784, 128])
        self.w2 = Array.randn([128,10])

    def forward(self, x):
        x1 = Array.matmul(x, self.w1)
        x2 = Array.maximum(x1, 0)
        return Array.matmul(x2, self.w2)
        
x = Array.randn([4,784]) 
net = FeedForwardNet()
y = net.forward(x)
print(list(y.shape))
