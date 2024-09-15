from lilgrad import *
import random
import numpy as np
from time import time

def init_weight(n,m):
    x = np.random.uniform(-1.,1.,size=(n*m))/np.sqrt(n*m).astype(np.float32)
    x = np.random.randn(n,m) * np.sqrt(2. / n)
    x = x.flatten()
    return Array(list(x), [n,m])

class Net:
    def __init__(self):
        self.w1 = Tensor(init_weight(784,128))
        self.w2 = Tensor(init_weight(128,10))

    def forward(self, x):
        x1 = matmul(x, self.w1)
        x2 = relu(x1)
        x3 = matmul(x2, self.w2)
        return logsoftmax(x3)

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255).astype(np.float32)
x_test = (x_test / 255).astype(np.float32)

model = Net()
print(Array.sum(model.w1.grad))
print(Array.sum(model.w2.grad))

lr = 0.005
BS = 128
losses = []

def test():
    test_batch = x_test.reshape(-1,28*28)
    test_batch_array = Array(list(test_batch.flatten()), shape=list(test_batch.shape))
    preds_out = model.forward(Tensor(test_batch_array))
    preds = np.argmax(np.array(list(preds_out.data.data)).reshape(preds_out.data.shape), axis=1)
    print('test acc:', np.mean(preds == y_test) * 100, '%')
    

for i in range(1000):
    samp = np.random.randint(0, x_train.shape[0], size=(BS))
    batch = x_train[samp].reshape(-1,28*28)
    y_samp = y_train[samp]
    shape = list(batch.shape)
    batch_array = Array(list(batch.flatten()), shape=shape)
    x = Tensor(batch_array)
    y = np.zeros((len(samp),10), np.float32)
    y[range(y.shape[0]), y_samp] = -1.0
    y_arr = Array(list(y.flatten()), list(y.shape))
    y = Tensor(y_arr)
    stime = time()
    out = model.forward(x)
    out_mul_y = mul(out, y)
    loss = mean(out_mul_y)
    loss.backward()
    batch_time = time() - stime
    model.w1.data = model.w1.data - model.w1.grad * lr
    model.w2.data = model.w2.data - model.w2.grad * lr
    model.w1.grad = Array.zeros_like(model.w1.grad)
    model.w2.grad = Array.zeros_like(model.w2.grad)
    if i % 100 == 0:
        print('batch', i, 'loss', loss.data, 'batch time', batch_time)

test()
