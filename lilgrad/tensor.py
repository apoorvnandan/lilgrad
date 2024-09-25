import ctypes
import numpy as np  # only to support creating tensors from numpy arrays
from ctypes import POINTER, c_float, c_int
arr_lib = ctypes.CDLL('./ops.so')

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
        return self.data.contents[key]  

    def convert_np(self, data):
        shape = (c_int * data.ndim)(*list(data.shape))
        ndim = len(shape)
        new_arr = arr_lib.create_arr(shape, ndim)
        data = data.flatten()
        for i in range(new_arr.contents.size):
            new_arr.contents.data[i] = data[i]
        return new_arr

    def backward(self):
        if self.op == 'relu':
            arr_lib.relu_backward(self.parents[0].grad, self.grad)
            self.parents[0].backward()
        elif self.op == 'matmul':
            arr_lib.matmul_backward(
                self.parents[0].grad, 
                self.parents[1].grad, 
                self.grad, 
                self.parents[0].data, 
                self.parents[1].data
            )
            self.parents[0].backward(grad_preset=True)
            self.parents[1].backward(grad_preset=True)
        elif self.op == 'nll_loss':
            arr_lib.nll_loss_backward(self.parents[0].grad, self.parents[1])
            self.parents[0].backward()
        elif self.op == 'logsoftmax':
            arr_lib.logsoftmax_backward(self.parents[0].grad, self.grad, self.data)
            self.parents[0].backward()
        
if __name__ == "__main__":
    data = np.arange(10).reshape(5,2)
    t = Tensor(data)
    for i in range(5):
        for j in range(2):
            print(i, j, t[i,j])
    
