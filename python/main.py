import random
import numpy as np
from utils import optimization as opti
from utils import activation_fun as acti
"""
X1 = np.array ([0,0,1,1]) # set of sample for the first input
X2 = np.array ([0,1,0,1]) # set of sample for the second input
Y = np.array ([0,0,0,1]) # set of expected result
"""

X1 = np.array ([-1,-1,1,1]) # set of sample for the first input
X2 = np.array ([-1,1,-1,1]) # set of sample for the second input
Y = np.array ([-1,-1,-1,1]) # set of expected result

nmax = 100
h = 0.01
n_sample = 4

"""
return the observed return value of the model with the parameters [w1 w2 b]
X1 X2 must be numpy array so that the function returns no error
"""
def linear_model (w1, w2, b, x1, x2) : 
    return w1*x1+w2*x2+b

"""
    - X known data (training data)
    - Y expected result
    returns the optimal parameters
"""
def training (X1, X2, Y) : 
    w1 = random.random()*10
    w2 = random.random()*10
    b = random.random()*10
    print("les données aléatoires sont ", w1, w2, b)


    return opti.gradient_descent([w1, w2, b], nmax, h, lambda C: opti.dhingeloss(C, n_sample, X1, X2, Y))

"""
returns the result of the perceptron for entry x1 and x2. The perceptron is trained for data X1, X2, Y 
"""
def perceptron (X1, X2, Y, x1, x2) : 
    w1, w2, b = training(X1, X2, Y)
    print("les données après training sont",w1, w2, b)
    z = linear_model(w1,w2, b, x1, x2)
    print("z vaut", z)
    return acti.step_fun(z)

print(perceptron (X1, X2, Y, 1, 0))