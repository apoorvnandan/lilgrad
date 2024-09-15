import ctypes
from time import time
import random
import math
import os

def load_cuda_lib():
    cudalib.cpu_to_cuda.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int]
    cudalib.cpu_to_cuda.restype = None

    cudalib.cuda_to_cpu.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int]
    cudalib.cuda_to_cpu.restype = None

    cudalib = ctypes.CDLL('./cudaops.so')
    cudalib.matmul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
    ]
    cudalib.matmul.restype = None
    return cudalib

cudalib = None
if os.path.exists('./cudaops.so'):
    cudalib = load_cuda_lib()

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

lib.transpose.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_int), 
]
lib.transpose.restype = None

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


lib.logfloat.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.logfloat.restype = None

lib.expfloat.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.expfloat.restype = None

lib.check_bool.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_char
]
lib.check_bool.restype = None

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

lib.sum_reduce.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
]
lib.sum_reduce.restype = None

lib.max_reduce.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
]
lib.max_reduce.restype = None

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
            if len(data) > 1:
                return (ctypes.c_float * len(data))(*data)
            return ctypes.pointer(ctypes.c_float(data[0]))
        elif dtype == 'int':
            if len(data) > 1:
                return (ctypes.c_int * len(data))(*data)
            return ctypes.pointer(ctypes.c_int(data[0]))
    else: # todo - add check for ctype array
        return data

def create_strides_from_shape(shape, itemsize=1):
    " shape is expected to be a list here "
    try: 
        if isinstance(shape, ctypes.Array):
            shape_list = list(shape)
        if isinstance(shape, ctypes._Pointer):
            shape_list = [shape.contents.value]
        if isinstance(shape, list):
            shape_list = shape
        strides = [itemsize]
        for s in reversed(shape_list[1:]):
            strides.append(strides[-1] * s)
        return list(reversed(strides))
    except:
        print(shape, type(shape))
        assert False

def lenc(x):
    if isinstance(x, ctypes.Array):
        return len(x)
    if isinstance(x, ctypes._Pointer):
        return 1
    if isinstance(x, list):
        return len(x)
    print(type(x))
    raise NotImplemented

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
        self.ndim = lenc(shape)
        self.size = lenc(data)
        self.device = 'cpu'

    def cuda(self):
        if not cudalib:
            raise ValueError('cuda not available')
        if self.device != 'cpu':
            return
        a_ptr = ctypes.cast(self.data, ctypes.POINTER(ctypes.c_float))
        a_ptr_ptr = ctypes.pointer(a_ptr)
        cudalib.cpu_to_cuda(a_ptr_ptr, self.size)
        self.device = 'cuda'

    def cpu(self):
        if not cudalib:
            raise ValueError('cuda not available')
        if self.device != 'cuda':
            return
        a_ptr = ctypes.cast(self.data, ctypes.POINTER(ctypes.c_float))
        a_ptr_ptr = ctypes.pointer(a_ptr)
        cudalib.cuda_to_cpu(a_ptr_ptr, self.size)
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
        if isinstance(other, (int, float)):
            other = Array(ctypes.pointer(ctypes.c_float(other)), [1])
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

    def check_bool(self, other, op):
        if isinstance(other, (int, float)):
            d = (ctypes.c_float * self.size)()
            lib.check_bool(self.data, d, self.size, other, op)
            return Array(d, self.shape)
        raise NotImplemented

    def __lt__(self, other): return self.check_bool(other, b'<')

    def __gt__(self, other): return self.check_bool(other, b'>')

    def __eq__(self, other): return self.check_bool(other, b'=')
 
    def transpose(self):
        d = (ctypes.c_float * self.size)()
        new_shape_list = list(reversed(list(self.shape)))
        new_shape = create_c_compat_array(new_shape_list, 'int')
        new_strides_list = create_strides_from_shape(new_shape_list)
        new_strides = create_c_compat_array(new_strides_list, 'int')
        lib.transpose(self.data, self.shape, self.strides, self.ndim, self.size, d, new_shape, new_strides)
        return Array(d, list(reversed(self.shape)))

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
        d_ptr = ctypes.cast(c_ctype, ctypes.POINTER(ctypes.c_float))
        d_ptr_ptr = ctypes.pointer(d_ptr)
        cudalib.cpu_to_cuda(d_ptr_ptr, size_result)
        if a.device == 'cuda' and b.device == 'cuda':
            if a.ndim != 2 and b.ndim != 2:
                raise ValueError(f'cuda matmul not supported for shapes {list(a.shape)} and {list(b.shape)}')
            cudalib.matmul(a, a.shape, b, b.shape, d)  
            return Array(d, shape)
         
        err = lib.matmul(
                a.data, a.shape, a.strides, a.ndim,
                b.data, b.shape, b.strides, b.ndim,
                d, shape, strides, len(shape_result)
        )
        if err == 1:
            assert False
        return Array(d, shape)

    def __str__(self): 
        if isinstance(self.data, ctypes._Pointer):
            return str(self.data.contents.value)
        return str(list(self.data))

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
    def abs(x):
        """
        doing this in python for testing
        will move to c later for speed
        """
        d = (ctypes.c_float * x.size)()
        for i in range(x.size):
            d[i] = abs(x.data[i])
        return Array(d, x.shape)

    @staticmethod
    def sum(x, axis=None):
        if axis is None:
            d = ctypes.pointer(ctypes.c_float(0))
            lib.sum(x.data, d, x.size)
            return Array(d, [1])
        shape_result = [x.shape[i] if i != axis else 1 for i in range(x.ndim)]
        size_result = 1
        for s in shape_result:
            size_result *= s
        d = (ctypes.c_float * size_result)()
        strides_result = create_strides_from_shape(shape_result)
        shape = create_c_compat_array(shape_result, 'int')
        strides = create_c_compat_array(strides_result, 'int')
        lib.sum_reduce(
                x.data, x.shape, x.strides, x.ndim,
                d, shape, strides, axis
        )
        return Array(d, shape)
    
    @staticmethod
    def mean(x):
        d = ctypes.pointer(ctypes.c_float(0))
        lib.mean(x.data, d, x.size)
        return Array(d, [1])

    @staticmethod
    def max(x, axis):
        shape_result = [x.shape[i] if i != axis else 1 for i in range(x.ndim)]
        size_result = 1
        for s in shape_result:
            size_result *= s
        d = (ctypes.c_float * size_result)()
        strides_result = create_strides_from_shape(shape_result)
        shape = create_c_compat_array(shape_result, 'int')
        strides = create_c_compat_array(strides_result, 'int')
        lib.max_reduce(
                x.data, x.shape, x.strides, x.ndim,
                d, shape, strides, axis
        )
        return Array(d, shape)

    @staticmethod
    def maximum(x, value):
        d = (ctypes.c_float * x.size)()
        lib.maximum(x.data, d, value, x.size)
        return Array(d, x.shape)

    def reshape(self, new_shape):
        if -1 in new_shape:
            remaining = 1
            for s in new_shape:
                if s != -1:
                    remaining *= s
            for i in range(len(new_shape)):
                if new_shape[i] == -1:
                    new_shape[i] = int(self.size/remaining)
        return Array(self.data, new_shape)


class Tensor:
    def __init__(self, data, name=''):
        self.data = data  # instance of array
        self.grad = Array.zeros_like(self.data)
        self.parents = []
        self.op = ''
        self.name = name

    def cuda(self): self.data.cuda()
    
    def cpu(self): self.data.cpu()

    @staticmethod
    def randn(shape):
        data = Array.randn(shape)
        return Tensor(data)

    def reshape(self, new_shape): return Tensor(self.data.reshape(new_shape))

    def backward(self, grad=None):
        if grad is None:
            grad = Array.ones_like(self.data)
        self.grad += grad
        if self.op == 'sum':
            assert len(self.parents) == 1
            self.parents[0].backward(Array.ones_like(self.parents[0].data) * grad)
        elif self.op == 'add':
            assert len(self.parents) == 2
            self.parents[0].backward(grad)
            self.parents[1].backward(grad)
        elif self.op == 'mul':
            assert len(self.parents) == 2
            self.parents[0].backward(grad * self.parents[1].data)
            self.parents[1].backward(grad * self.parents[0].data)
        elif self.op == 'relu':
            assert len(self.parents) == 1
            mask = (self.data > 0)
            self.parents[0].backward(grad * mask)
        elif self.op == 'mean':
            assert len(self.parents) == 1
            k = 1.0 / self.data.size
            self.parents[0].backward(grad * k)
        elif self.op == 'logsoftmax':
            assert len(self.parents) == 1
            softmax_output = self.data.exp()
            x = Array.sum(grad, axis=1)
            x1 = x.reshape([-1,1])
            grad_out = grad - softmax_output * x1
            self.parents[0].backward(grad_out)
        elif self.op == 'matmul':
            assert len(self.parents) == 2
            t0 = self.parents[0].data.transpose()
            t1 = self.parents[1].data.transpose()
            self.parents[0].backward(Array.matmul(grad, t1))
            self.parents[1].backward(Array.matmul(t0, grad))


def add(a, b):
    c = Tensor(a.data + b.data)
    c.parents = [a, b]
    c.op = 'add'
    return c

def mul(a, b):
    c = Tensor(a.data * b.data)
    c.parents = [a, b]
    c.op = 'mul'
    return c

def relu(a):
    b = Tensor(Array.maximum(a.data, 0))
    b.op = 'relu'
    b.parents = [a]
    return b

def mean(a):
    c = Tensor(Array.mean(a.data))
    c.parents = [a]
    c.op = 'mean'
    return c

def logsoftmax(a):
    max_vals = Array.max(a.data, axis=1)
    diff = (a.data - max_vals)
    exp_a = diff.exp()
    sum_exp_a = Array.sum(exp_a, axis=1)
    z = sum_exp_a.log()
    c = Tensor(diff - z)
    c.parents = [a]
    c.op = 'logsoftmax'
    return c 

def matmul(a, b):
    c = Tensor(Array.matmul(a.data, b.data))
    c.parents = [a,b]
    c.op = 'matmul'
    return c


