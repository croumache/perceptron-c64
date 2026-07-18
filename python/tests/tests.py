import utils.optimization as opti
import numpy as np

X1 = np.array ([-1,-1,1,1]) # set of sample for the first input
X2 = np.array ([-1,1,-1,1]) # set of sample for the second input
Y = np.array ([-1,-1,-1,1]) # set of expected result

nmax = 10000
h = 0.001
tol = 0.001
n_sample = 4

"""
Testing the gradient descent with :

- f(x, y) = x**2 + (y-1)**2, the minimum is at (0, 1)


"""

print("Descente du gradient", opti.gradient_descent([0, 0], tol, nmax, h, lambda C: np.array([2*C[0], 2*(C[1] - 1)])))

#python3 -m tests.tests (in the python directory)