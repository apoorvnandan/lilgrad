# Test the implementation by training on MNIST
from lilgrad import Tensor, zeros, matmul, relu, logsoftmax, nll_loss
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = (x_train / 255).astype(np.float32)
x_test = (x_test / 255).astype(np.float32)

def uniform_init_matrix(n,m):
    return np.random.uniform(-1., 1., size=(n,m))/np.sqrt(n*m).astype(np.float32)

class TinyNet:
    def __init__(self):
        self.w1 = Tensor(uniform_init_matrix(784,128))
        self.w2 = Tensor(uniform_init_matrix(128,10))

    def set_batch_size(self, B):
        " preallocates all the needed arrays "
        self.w1_out = zeros((B, 128))
        self.relu_out = zeros((B, 128))
        self.w2_out = zeros((B,10))
        self.logsoftmax_out = zeros((B,10))

    def forward(self, x):
        matmul(self.w1_out, x, self.w1)
        relu(self.relu_out, self.w1_out)
        matmul(self.w2_out, self.relu_out, self.w2)
        logsoftmax(self.logsoftmax_out, self.w2_out)
        return self.logsoftmax_out


model = TinyNet()
lr = 0.005
BS = 128
model.set_batch_size(128)
loss = zeros([1])


for i in range(1000):
    samp = np.random.randint(0, x_train.data.shape[0], size=(BS))
    x = Tensor(x_train[samp].reshape(-1,28*28))
    y_samp = y_train[samp]
    y = np.zeros((len(samp),10), np.float32)
    y[range(y.shape[0]),y_samp] = 1.0
    y = Tensor(y)

    out = model.forward(x)
    nll_loss(loss, out, y)
    loss.backward()

    if i % 100 == 0:
        print('batch',i,'loss',loss[0])

    model.w1.update_grad(lr)
    model.w2.update_grad(lr)

    model.w1.zero_grad()
    model.w2.zero_grad()

