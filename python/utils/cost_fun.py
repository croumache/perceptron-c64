import math

N = 20 # nombre de terme dans le raisonnement de Taylor

def logloss(x, y, n) :
    """
    x données obtenues et y ce à quoi on s'attend
    x est calculée en amont par la combinaison linéaire
    y ne prend que les valeurs 1 et 0
    """
    result = 0
    for i in range(n) :
        result +=  x[i]*math.log(y[i]) + (1-x[i])*math.log((1-y[i]))
    return -1/n * result 

def log_taylor(x) : 
    result = 0
    for i in range (1,N+1) : 
        result +=((-1)**(i+1))*((x-1)**i)/i
    return result


def logloss_taylor(x,y,n) : 
    result = 0 
    for i in range(n) : 
        result += x[i]*log_taylor(y[i]) + (1-x[i])*log_taylor((1-y[i]))
    return -1/n *result 

def hingeloss(x,y,n) :
    sum = 0
    for i in range(n) :
        sum = sum + max(0, 1-x[i]*y[i])
    return sum
