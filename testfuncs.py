import numpy as np
import matplotlib.pyplot as plt

def fibonacci1(N):
    """
    Prints the fibonacci numbers(until the n'th) and returns a numpy vector
    containing those numbers.
    """
    if N == 0:
        print(0)
    fibvals = np.zeros((2,1))
    fibvals[0] = 1
    fibvals[1] = 1
    for i in range(N):
        print(fibvals[i%2])
        fibvals[i%2] = fibvals[i%2] + fibvals[(i+1)%2]
    return fibvals

def fibonacci2(N):
    if N == 0:
        return 0
    if N == 1 or N == 2:
        return 1
    return fibonacci2(N-1) + fibonacci2(N-2)

def heron(a,x0,tol=1e-14):
    maxit = 25
    if a < 0 or x0 <= 0:
        print('Incorrect inputs')
        return
    xold = x0
    for i in range(maxit):
        x= (1/2)*(xold + a/xold)
        if abs(x - xold) <= tol:
            print("Tolerance attained with %d iterations and a value of %f" % (i,x))
            return x
        xold = x
    print('Max Iterations attained!')
    return 0

def listRegression(x,y):
    xmean = sum(x)/len(x)
    ymean = sum(y)/len(y)
    num = 0
    denom = 0
    for i in range(len(x)):
        num += (x[i] - xmean)*(y[i] - ymean)
        denom += (x[i] - xmean)**2
    alpha = num/denom
    beta = ymean - alpha*xmean
    return alpha,beta

def numpRegression(x,y):
    xmean = np.mean(x)
    ymean = np.mean(y)
    alpha = np.mean((x - xmean)*(y - ymean))/np.mean((x - xmean)**2)
    beta = ymean - alpha*xmean
    return alpha, beta