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

# Get the train_batch function from the library
train_batch = cuda_lib.train_batch

# Define the argument types for train_batch
train_batch.argtypes = [
    ctypes.POINTER(Arr),  # inp
    ctypes.POINTER(Arr),  # labels
    ctypes.POINTER(Arr),  # w1
    ctypes.POINTER(Arr),  # w2
    ctypes.c_float,       # lr
    ctypes.POINTER(Arr),  # w1_out
    ctypes.POINTER(Arr),  # relu_out
    ctypes.POINTER(Arr),  # w2_out
    ctypes.POINTER(Arr),  # logsoftmax_out
    ctypes.POINTER(Arr),  # loss
    ctypes.POINTER(Arr),  # logsoftmax_out_grad
    ctypes.POINTER(Arr),  # w2_out_grad
    ctypes.POINTER(Arr),  # w2_grad
    ctypes.POINTER(Arr),  # relu_out_grad
    ctypes.POINTER(Arr),  # w1_out_grad
    ctypes.POINTER(Arr)   # w1_grad
]

# Define the return type (void in this case)
train_batch.restype = None

def move_to_cuda(arr_list):
    for a in arr_list:
        cuda_lib.cpu_to_cuda(ctypes.byref(a))

def init_weight(n,m):
    return np.random.randn(n,m) * np.sqrt(2. / n)

def test_mnist():
    BS = 128
    lr = 0.005

    w1 = numpy_to_arr(init_weight(784,128))
    w2 = numpy_to_arr(init_weight(128,10))

    # Create output arrays (you'll need to initialize these with the correct shapes)
    w1_out = numpy_to_arr(np.zeros((BS, 128), dtype=np.float32))
    relu_out = numpy_to_arr(np.zeros((BS, 128), dtype=np.float32))
    w2_out = numpy_to_arr(np.zeros((BS, 10), dtype=np.float32))
    logsoftmax_out = numpy_to_arr(np.zeros((BS, 10), dtype=np.float32))
    loss = numpy_to_arr(np.zeros(1, dtype=np.float32))

    # Create gradient arrays
    logsoftmax_out_grad = numpy_to_arr(np.zeros((BS, 10), dtype=np.float32))
    w2_out_grad = numpy_to_arr(np.zeros((BS, 10), dtype=np.float32))
    w2_grad = numpy_to_arr(np.zeros((128, 10), dtype=np.float32))
    relu_out_grad = numpy_to_arr(np.zeros((BS, 128), dtype=np.float32))
    w1_out_grad = numpy_to_arr(np.zeros((BS, 128), dtype=np.float32))
    w1_grad = numpy_to_arr(np.zeros((784, 128), dtype=np.float32))

    move_to_cuda([w1_out, relu_out, w2_out, logsoftmax_out, loss, w1, w2, logsoftmax_out_grad,
        w2_out_grad, w2_grad, relu_out_grad, w1_out_grad, w1_grad])

    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255).astype(np.float32)
    x_test = (x_test / 255).astype(np.float32)
    for i in range(1000):
        samp = np.random.randint(0, x_train.shape[0], size=(BS))
        batch = x_train[samp].reshape(-1,28*28)
        y_samp = y_train[samp]
        y = np.zeros((len(samp),10), np.float32)
        y[range(y.shape[0]), y_samp] = -1.0
        inp = numpy_to_arr(batch)
        labels = numpy_to_arr(y)
        cuda_lib.cpu_to_cuda(ctypes.byref(inp))
        cuda_lib.cpu_to_cuda(ctypes.byref(labels))
        train_batch(
            ctypes.byref(inp), ctypes.byref(labels),
            ctypes.byref(w1), ctypes.byref(w2),
            ctypes.c_float(lr),
            ctypes.byref(w1_out), ctypes.byref(relu_out), ctypes.byref(w2_out),
            ctypes.byref(logsoftmax_out), ctypes.byref(loss),
            ctypes.byref(logsoftmax_out_grad), # logsoftmax_out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(w2_out_grad), # w2_out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(w2_grad), # w2_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(relu_out_grad), # relu_out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(w1_out_grad), # w1_out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(w1_grad) # w1_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        cuda_lib.cuda_to_cpu(ctypes.byref(inp));
        cuda_lib.cuda_to_cpu(ctypes.byref(labels));
        if i % 100 == 0:
            cuda_lib.cuda_to_cpu(ctypes.byref(loss));
            print('batch', i, 'loss', loss.data[0])
            cuda_lib.cpu_to_cuda(ctypes.byref(loss));

test_mnist()
