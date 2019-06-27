# normalizeRows
import numpy as np

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    Argument: x -- A numpy matrix of shape (n,m)
    Returns: x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x / x_norm
    print("shape of x_norm: " + str(x_norm.shape))
    print("shape of x: " + str(x.shape))
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))


