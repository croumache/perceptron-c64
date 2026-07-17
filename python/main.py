import random
from utils import optimization as opti

nmax = 100
h = 0.01

def linear_model (w, b, X) : 
    w1, w2 = w
    x1, x2 = X
    return w1*x1+w2*x2+b

"""
    - X known data
    - Y expected result
"""
def training (X, Y, n) : 
    w1 = random.rand()*10
    w2 = random.rand()*10
    b = random.rand()*10
    


    return opti.gradient_descent([w1, w2, b], nmax, h, lambda )

