def logloss (x,y,n) :
    return 1

def hingeloss(x,y,n) :
    sum = 0
    for i in range(n) :
        sum = sum + max(0, 1-x[i]*y[i])
    return sum
