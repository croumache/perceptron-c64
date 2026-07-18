import numpy as np


"""
return the observed return value of the model with the parameters [w1 w2 b]
X1 X2 must be numpy array so that the function returns no error
"""
def linear_model (w1, w2, b, x1, x2) : 
    return w1*x1+w2*x2+b

"""
- C0 is the initial parameters of the model (w1, w2, b)
- n number of iteration
- h step size
- df the gradient of the function 

return the optimal parameters 
"""
def gradient_descent(C0, tol, nmax, h, df) :
    n = 0
    C_prev = C0
    C = C_prev - h*df(C_prev)
    while(np.linalg.norm(C - C_prev) > tol) :
        
        if (n > nmax) :
            print("nmax has been reached")
            return C
        
        C_prev = C
        C = C_prev - h*df(C_prev)

        n = n + 1    
    return C

"""
derivative of hingeloss
- C = [w1, w2, b]
- n is the number of sample
- X1 is the set of samples for the first input
- X2 is the set of samples for the second input
- Y is the expected output for each pair of sample (x1, x2).

"""
def dhingeloss(C, n, X1, X2, Y) :
    w1, w2, b = C[0], C[1], C[2]
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(n):
        if(1-(Y[i]*linear_model(w1, w2, b, X1[i], X2[i])) > 0) :
            dw1 = dw1 - Y[i]*X1[i]
            dw2 = dw2 - Y[i]*X2[i]
            db = db - Y[i]
    return np.array([dw1, dw2, db])

"""
derivative of logloss
- C = [w1, w2, b]
- n is the number of sample
- X1 is the set of samples for the first input
- X2 is the set of samples for the second input
- Y is the expected output for each pair of sample (x1, x2).
"""

def dlogloss(C, n, X1, X2, Y) :
    w1, w2, b = C[0], C[1], C[2]
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(n):
        dw1 = X1[i]*( (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i]/(w1*X1[i] + w2*X2[i] + b - 1)) )
        dw2 = X2[i]*( (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i]/(w1*X1[i] + w2*X2[i] + b - 1)) )
        db = (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i])/(w1*X1[i] + w2*X2[i] + b - 1)
    return np.array([dw1, dw2, db])