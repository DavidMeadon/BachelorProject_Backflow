import numpy as np
import matplotlib.pyplot as plt

def fibonacci1(N):
    if N == 0:
        print(0)
    fibvals = np.zeros((2,1))
    fibvals[0] = 1
    fibvals[1] = 1
    for i in range(N):
        print(fibvals[i%2])
        fibvals[i%2] = fibvals[i%2] + fibvals[(i+1)%2]
    
fibonacci1(8)

def fibonacci2(N):
    if N == 0:
        return 0
    if N == 1 or N == 2:
        return 1
    return fibonacci2(N-1) + fibonacci2(N-2)

N = 8
for i in range(N):
    print(fibonacci2(i+1))

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

print(abs(heron(np.pi,1) - np.sqrt(np.pi)))

# Linear Regression

N = int(1e6)
x = []
y = []
for idx in range(N+1):
    x.append(idx/N)
    y.append(x[idx] + np.random.standard_normal(1))

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

alpha,beta = listRegression(x,y)

approxy = []
for j in range(len(x)):
    approxy.append(alpha*x[j] + beta)

# plt.plot(x,y)
# plt.plot(x,approxy)
# plt.plot(x,x)
# plt.show()

x2 = np.zeros((len(x),1))
y2 = np.zeros((len(y),1))

for idx in range(len(x)):
    x2[idx] = idx/N
    y2[idx] = x2[idx] + np.random.standard_normal(1)


def numpRegression(x,y):
    xmean = np.mean(x)
    ymean = np.mean(y)
    alpha = np.mean((x - xmean)*(y - ymean))/np.mean((x - xmean)**2)
    beta = ymean - alpha*xmean
    return alpha, beta

alpha2,beta2 = numpRegression(x2,y2)

# Using iPython more than 100 times faster

