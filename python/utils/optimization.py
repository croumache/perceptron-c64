"""
- C0 is the initial parameters of the model (w1, w2, b)
- n number of iteration
- h step size
- the gradient of the function 

return the optimal parameters 
"""
def gradient_descent(C0, nmax, h, df) :
    C_prev = C0
    C = C0 + h*df(C0)
    for i in range(nmax) :
        C  = C_prev + h*df(C0)
        C_prev = C
    return C

"""
derivative of hingeloss
- C = [w1, w2, b]
- n is the number of sample

"""
def dhingeloss(C, n, X1, X2, Y) : #la dérivée est constante => ne dépend pas de w1, w2 et b.
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(n):
        dw1 = dw1 + Y[i]*X1[i]
        dw2 = dw2 + Y[i]*X2[i]
        db = db + Y[i]
    return [dw1, dw2, db]

"""
derivative of logloss
"""

def dlogloss(w1, w2, b, n, X1, X2, Y) :
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(n):
        dw1 = X1[i]*( (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i]/(w1*X1[i] + w2*X2[i] + b - 1)) )
        dw2 = X2[i]*( (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i]/(w1*X1[i] + w2*X2[i] + b - 1)) )
        db = (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i])/(w1*X1[i] + w2*X2[i] + b - 1)
    return [dw1, dw2, db]