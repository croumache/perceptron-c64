import random
import numpy as np
from utils import optimization as opt
from utils import activation_fun as act

"""
set of training data
"""
X1 = np.array ([0,0,1,1]) # set of sample for the first input
X2 = np.array ([0,1,0,1]) # set of sample for the second input
Y = np.array ([0,0,0,1]) # set of expected result

"""
global constants
"""
nmax = 10000
h = 0.001
tol = 0.001
n_sample = 4


"""
- X known data (training data)
- Y expected result
returns the optimal parameters
"""

def training (X1, X2, Y) : 
    w1 = random.random()
    w2 = random.random()
    b = random.random()

    print("random initial parameters values : ", w1, w2, b)

    return opt.gradient_descent([w1, w2, b], tol, nmax, h, lambda C: opt.dhingeloss(C, n_sample, X1, X2, Y))

"""
- X known data (training data)
- Y expected result
- x entry
returns the result of the perceptron for entry x1 and x2. 
"""
def perceptron (X1, X2, Y, x1, x2) :
    X1 = 2*X1-1
    X2 = 2*X2-1
    Y = 2*Y-1 
    x1 = 2*x1-1
    x2 = 2*x2-1
    w1, w2, b = training(X1, X2, Y)
    print("parameters values after training : ",w1, w2, b)
    z = opt.linear_model(w1,w2, b, x1, x2)
    print("z =", z)
    return act.step_fun(z)

print(perceptron(X1, X2, Y, 0,1))
