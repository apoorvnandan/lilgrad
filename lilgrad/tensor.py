import ctypes
import numpy as np  # only to support creating tensors from numpy arrays
from ctypes import POINTER, c_float, c_int
arr_lib = ctypes.CDLL('./lilgrad/ops.so')

class Arr(ctypes.Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("shape", POINTER(c_int)),
        ("strides", POINTER(c_int)),
        ("ndim", c_int),
        ("size", c_int)
    ]

arr_lib.create_arr.argtypes = [POINTER(c_int), c_int]
arr_lib.create_arr.restype = POINTER(Arr)
arr_lib.zeros_like.argtypes = [POINTER(Arr)]
arr_lib.zeros_like.restype = POINTER(Arr)
arr_lib.free_arr.argtypes = [POINTER(Arr)]
arr_lib.free_arr.restype = None

# matmul and its backward function
arr_lib.matmul.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr)]
arr_lib.matmul.restype = None
arr_lib.matmul_backward.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr), POINTER(Arr), POINTER(Arr)]
arr_lib.matmul_backward.restype = None

# relu and relu_backward
arr_lib.relu.argtypes = [POINTER(Arr), POINTER(Arr)]
arr_lib.relu.restype = None
arr_lib.relu_backward.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr)]
arr_lib.relu_backward.restype = None

# logsoftmax and its backward
arr_lib.logsoftmax.argtypes = [POINTER(Arr), POINTER(Arr)]
arr_lib.logsoftmax.restype = None
arr_lib.logsoftmax_backward.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr)]
arr_lib.logsoftmax_backward.restype = None

# lossfn and its backward
arr_lib.nll_loss.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr)]
arr_lib.nll_loss.restype = None
arr_lib.nll_loss_backward.argtypes = [POINTER(Arr), POINTER(Arr)]
arr_lib.nll_loss_backward.restype = None

# conv2d
arr_lib.conv2d.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr)]
arr_lib.conv2d.restype = None

# maxpool2d
arr_lib.maxpool2d.argtypes = [POINTER(Arr), POINTER(Arr), c_int, c_int]
arr_lib.maxpool2d.restype = None

# conv2d_backward
arr_lib.conv2d_backward.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr), POINTER(Arr), POINTER(Arr)]
arr_lib.conv2d_backward.restype = None

# maxpool2d_backward
arr_lib.maxpool2d_backward.argtypes = [POINTER(Arr), POINTER(Arr), POINTER(Arr), c_int, c_int]
arr_lib.maxpool2d_backward.restype = None

# view
arr_lib.view.argtypes = [POINTER(Arr), POINTER(c_int), c_int]
arr_lib.view.restype = None

# view_backward
arr_lib.view_backward.argtypes = [POINTER(Arr), POINTER(c_int), c_int]
arr_lib.view_backward.restype = None

# update weights
arr_lib.update_grad.argtypes = [POINTER(Arr), POINTER(Arr), c_float]
arr_lib.update_grad.restype = None

# set to zero
arr_lib.set_zero.argtypes = [POINTER(Arr)]
arr_lib.set_zero.restype = None


class Tensor:
    def __init__(self, data):
        self.data = self.convert_np(data)
        self.grad = arr_lib.zeros_like(self.data)
        self.op = ''
        self.parents = []

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != self.data.contents.ndim:
                raise ValueError(f"indices must have {self.data.contents.ndim} dimensions")
            pos = 0
            for i in range(self.data.contents.ndim):
                stride = self.data.contents.strides[i]
                pos += stride * key[i]
            return self.data.contents.data[pos]
        return self.data.contents.data[key]
    
    def zero_grad(self):
        arr_lib.set_zero(self.grad)

    def convert_np(self, data):
        if isinstance(data, POINTER(Arr)):
            return data
        shape = (c_int * data.ndim)(*list(data.shape))
        ndim = len(shape)
        new_arr = arr_lib.create_arr(shape, ndim)
        data = data.flatten()
        for i in range(new_arr.contents.size):
            new_arr.contents.data[i] = data[i]
        return new_arr

    def update_grad(self, lr):
        arr_lib.update_grad(self.data, self.grad, lr)

    def backward(self):
        if self.op == 'relu':
            arr_lib.relu_backward(self.parents[0].grad, self.grad, self.parents[0].data)
            self.parents[0].backward()
        elif self.op == 'matmul':
            arr_lib.matmul_backward(
                self.parents[0].grad, 
                self.parents[1].grad, 
                self.grad, 
                self.parents[0].data, 
                self.parents[1].data
            )
            self.parents[0].backward()
            self.parents[1].backward()
        elif self.op == 'nll_loss':
            arr_lib.nll_loss_backward(self.parents[0].grad, self.parents[1].data)
            self.parents[0].backward()
        elif self.op == 'logsoftmax':
            arr_lib.logsoftmax_backward(self.parents[0].grad, self.grad, self.data)
            self.parents[0].backward()

def matmul(c, a, b):
    arr_lib.matmul(c.data, a.data, b.data)
    c.op = 'matmul'
    c.parents = [a,b]

def relu(out, inp):
    arr_lib.relu(out.data, inp.data)
    out.op = 'relu'
    out.parents = [inp]

def logsoftmax(out, inp):
    arr_lib.logsoftmax(out.data, inp.data)
    out.op = 'logsoftmax'
    out.parents = [inp]

def nll_loss(loss, out, labels):
    arr_lib.nll_loss(loss.data, out.data, labels.data)
    loss.op = 'nll_loss'
    loss.parents = [out, labels]

def zeros(shape):
    s = (c_int * len(shape))(*list(shape))
    nd = len(shape)
    d = arr_lib.create_arr(s, nd)
    return Tensor(d)
        
if __name__ == "__main__":
    data = np.arange(10).reshape(5,2)
    t = Tensor(data)
    for i in range(5):
        for j in range(2):
            print(i, j, t[i,j])
    
