# Creating a tiny tensor library in C

Documenting my pov while building this. The objective is to help people understand neural networks from absolute scratch. No pytorch. No numpy. 

Should be readable for anyone who knows programming (any language) even if you are not familiar with machine learning or neural networks at all. 

Moreover, building this from scratch, and in C, does not mean that the code and the APIs will not be user friendly. In fact, we'll create the needed abstractions and show just how easy it is to code and train different neural network architectures with this tiny library.

## Contents:

* What are neural networks anyhow?
* Creating tensors
* Defining a loss function
* Optimising the loss - Autograd
* Examples of training different neural networks

## What are neural networks anyhow?

Think about the process of coding a function. For some tasks, functions can be straightforward to code.

Here's an example: "write a function to change this color image to grayscale".

There is a clear set of instructions (for each pixel: val = 0.299R + 0.587G + 0.114B) that you can code in your favorite programming language and create a function that will solve this task. The function will be determinisitic, giving you the exact same grayscale image for the same color image.

There are other tasks, where it's pretty much impossible to come up with a set of instructions needed to get the output from the input. And therefore, you cannot write the code needed for the function.

Example: "The input image contains either a cat or a dog. Write a function to output `CAT` or `DOG` based on the image contents.".

Think about the code you can write for this. You'll quickly realise that there is no specifi set of instructions you can code here, to create this function. You can, however, write a special kind of a function that can solve this task. Let's write one such special function. The code is in C, but the logic should be readable by anyone.

```c
int cat_or_dog(float* input_img, float* w1, float* w2) {
    float* x1 = matrix_multiplication(input_img, w1);
    float* x2 = relu(x1);
    float* x3 = matrix_multiplication(x2, w2);
    float* x4 = logsoftmax(x3);
    if (x4[0] < x4[1]) return 0; // indiciating a "cat"
    return 1; // indicating a "dog"
}
```

You'll notice that this function is a bit weird. First of all it takes two other float arrays `w1` and `w2` as inputs, apart from the image. (The image here is a grayscale picture where each pixel value (0-255) is divided by 255 to represent it as a large float array)

Then, the function proceeds to do some weird mathematical operations on the image.

* `matrix_multiplication` is self explanatory.
* `relu` is a mathematical operation, equivalent of max(x, 0) on every number within the array.
* `logsoftmax` is a another mathematical function, and I'll just write the formula below, because it's not important for the point I'm trying to make at the moment.

The weirdest thing though, is that this function, for very specific values of `w1` and `w2`, will actually give you the correct output for like 99% of images. If you're completely new to machine learning and neural networks, you might be surprised. Infact, I'll link the exact values of w1 and w2 along with the code, so you can test this on a bunch of cat and dog images of a certain size.

Functions like these are called neural networks. And the additional inputs like `w1` and `w2` are called parameters or weights. When you "train" a neural network, you find the "correct" values of these parameters for the task you're trying to solve. 

You've probably heard about, or used ChatGPT, and other similar AI assistants. They are powered by neural networks as well. They generate their response to your message, by converting your text into some numbers, and then doing these weird mathematical operations between them and the parameters of the network to output the next word (or part of the word). Then they do this until the entire response is generated! Anyhow, we'll do something similar from scratch and build a nice little library to code different neural networks for different tasks, and train them.

## Creating tensors

Okay, I'll assume you are familiar with 1D arrays, 2D arrays, etc. 
We will work with N-D arrays here. And we will call these N-D arrays tensors.

To work with N-D arrays, we will create a `struct` called `Arr` here.

```c
typedef struct {
    float* values;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;
```

This struct holds everything needed. All the values are inside the 1D values array. Shape should be obvious. Strides is something that's really useful for some operations, and I'll explain them later. 

Here's an example:
To do matrix multiplication between two 2D arrays with shapes (4,2) and (2,3), we can write the following code.

```
...
...
```

## Defining a loss function

Going back to the initial example, our input, along with the parameters `w1` and `w2` are all instances of tensors. The challenge here is to find those specific values for `w1` and `w2` which make the function actually work.

In order to do that, we define a different function, one that operates on the outputs of our neural network function and the expected outputs and returns a score that represents how good or bad the neural network is.

Here is an example:

```c
nll_loss
```

To understand what this function does, you should 
