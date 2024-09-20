import ctypes
import numpy as np

# Load the shared library
nn_lib = ctypes.CDLL('./nn.so')

# Define the Arr struct in Python
class Arr(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int)
    ]

# Get the train_batch function from the library
train_batch = nn_lib.train_batch

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
    ctypes.POINTER(ctypes.c_float),  # logsoftmax_out_grad
    ctypes.POINTER(ctypes.c_float),  # w2_out_grad
    ctypes.POINTER(ctypes.c_float),  # w2_grad
    ctypes.POINTER(ctypes.c_float),  # relu_out_grad
    ctypes.POINTER(ctypes.c_float),  # w1_out_grad
    ctypes.POINTER(ctypes.c_float)   # w1_grad
]

# Define the return type (void in this case)
train_batch.restype = None

def numpy_to_ctype(np_array):
    data = list(np_array.flatten())
    data_array = (ctypes.c_float * np_array.size)(*data)
    return ctypes.cast(data_array, ctypes.POINTER(ctypes.c_float))
    
# Helper function to create an Arr from a numpy array
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
    logsoftmax_out_grad = np.zeros((BS, 10), dtype=np.float32)
    w2_out_grad = np.zeros((BS, 10), dtype=np.float32)
    w2_grad = np.zeros((128, 10), dtype=np.float32)
    relu_out_grad = np.zeros((BS, 128), dtype=np.float32)
    w1_out_grad = np.zeros((BS, 128), dtype=np.float32)
    w1_grad = np.zeros((784, 128), dtype=np.float32)

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
        train_batch(
            ctypes.byref(inp), ctypes.byref(labels),
            ctypes.byref(w1), ctypes.byref(w2),
            ctypes.c_float(lr),
            ctypes.byref(w1_out), ctypes.byref(relu_out), ctypes.byref(w2_out),
            ctypes.byref(logsoftmax_out), ctypes.byref(loss),
            numpy_to_ctype(logsoftmax_out_grad), # logsoftmax_out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            numpy_to_ctype(w2_out_grad), # w2_out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            numpy_to_ctype(w2_grad), # w2_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            numpy_to_ctype(relu_out_grad), # relu_out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            numpy_to_ctype(w1_out_grad), # w1_out_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            numpy_to_ctype(w1_grad) # w1_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if i % 100 == 0:
            print('batch', i, 'loss', loss.data[0])


test_mnist()
