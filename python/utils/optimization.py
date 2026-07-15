def gradient_descent(W0, n, h, df) :
    W_prev = W0
    W = W0 + h*df(W0)
    for i in range(n) :
        W  = W_prev + h*df(W0)
        W_prev = W
    return W

def dhingeloss(n, X1, X2, Y) : #la dérivée est constante => ne dépend pas de w1, w2 et b.
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(n):
        dw1 = dw1 + Y[i]*X1[i]
        dw2 = dw2 + Y[i]*X2[i]
        db = db + Y[i]
    return [dw1, dw2, db]

def dlogloss(w1, w2, b, n, X1, X2, Y) :
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(n):
        dw1 = X1[i]*( (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i]/(w1*X1[i] + w2*X2[i] + b - 1)) )
        dw2 = X2[i]*( (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i]/(w1*X1[i] + w2*X2[i] + b - 1)) )
        db = (Y[i]/(w1*X1[i] + w2*X2[i] + b)) + (1-Y[i])/(w1*X1[i] + w2*X2[i] + b - 1)
    return [dw1, dw2, db]