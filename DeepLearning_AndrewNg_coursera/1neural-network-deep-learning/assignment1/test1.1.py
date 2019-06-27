test = "Hello World"
print("test: " + test)

import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.
    Arguments: x -- A scalar
    Return: s -- sigmoid(x)
    """
    s = 1 / (1 + math.exp(-x))

    return s


print(basic_sigmoid(3))


import numpy as np

# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x))
print(x+3)
print(1/x)

def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments: x -- A scalar or numpy array of any size
    Return: s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

x = np.array([1,2,3])
print(sigmoid(x))
