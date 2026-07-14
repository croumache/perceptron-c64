import math

N = 20 # number of terms in taylor series

#implementation of the step function
def step_fun(x) : 
    if x >= 0 : 
        return 1
    else :  
        return 0

#implementation of the sigmoid function whith an exponential
def sigmoid(x) : 
    return 1/(1+math.exp(-x))

#implementation of a taylor serie for exponential
def exp_taylor(x) :
    result = 0 
    for i in range(N) : 
        result += x**i/math.factorial(i)
    return result

#implementation of a sigmoid with a taylor serie for the exponential
def sigmoid_Taylor(x) : 
    return 1/(1+exp_taylor(-x))

