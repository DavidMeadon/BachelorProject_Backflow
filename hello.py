import numpy as np
import matplotlib.pyplot as plt


import sys

iMaxStackSize = 5000
sys.setrecursionlimit(iMaxStackSize)



# x = np.arcsin(1)

# y = np.zeros((5,1))
# y[0] = 0
# y[1] = 1
# y[2] = 1
# y[3] = 2
# y[4] = 3
# plt.plot(y)

# print(x, np.pi/2)
# plt.show()
def Dartspi(M):
    hit = 0
    throw = np.zeros((2,1))
    for k in range(M):
        throw[0] = np.random.rand(1)
        throw[1] = np.random.rand(1)
        if np.linalg.norm(throw) < 1:
            hit += 1
    piApprox = 4*hit/M
    return piApprox
# print('%.15f' % piApprox)

def piNewton(n,k):
    if n > k:
        return 1
    y = 1 + (n/(2*n+1))*piNewton(n+1,k)
    return y

def circumscrib(k):
    if k == 6:
        return 4*np.sqrt(3)
    y = 2*inscrib(k/2)*circumscrib(k/2)/(inscrib(k/2) + circumscrib(k/2))
    return y

def inscrib(k):
    if k == 6:
        return 6
    y = np.sqrt(inscrib(k/2)*circumscrib(k))
    return y

# temp = 3*(2**10)

# piApprox2 = (circumscrib(temp) + inscrib(temp))/4

# print(piApprox2)



def Baselpi(n):
    sol = 0
    for y in range(n):
        sol += 1/((y+1)**2)
    return sol

def Pinspi(n):
    hit = 0
    for k in range(n):
        xpos = np.random.rand(1)
        angle = 2*np.pi*np.random.rand(1)
        newxpos1 = xpos + np.cos(angle)*0.25
        newxpos2 = xpos - np.cos(angle)*0.25
        if newxpos1 >=1 or newxpos1 <=0 or newxpos2 >=1 or newxpos2 <=0:
            hit += 1
    if hit == 0:
        return 1
    return n/hit


h = 15
error = np.zeros((h,1))
error2 = np.zeros((h,1))
error3 = np.zeros((h,1))
error4 = np.zeros((h,1))
error5 = np.zeros((h,1))
for k in range(h):
    j = 2**k
    u = 6*2**k
    error[k] = abs(np.pi - 2*piNewton(1,j))
    piApprox2 = (circumscrib(u) + inscrib(u))/4
    error2[k] = abs(np.pi - piApprox2)
    error3[k] = abs(np.pi - np.sqrt(6*Baselpi(j)))
    error4[k] = abs(np.pi - Dartspi(j))
    error5[k] = abs(np.pi - Pinspi(j))
# plt.plot(<X AXIS VALUES HERE>, <Y AXIS VALUES HERE>, 'line type', label='label here')
# plt.plot(<X AXIS VALUES HERE>, <Y AXIS VALUES HERE>, 'line type', label='label here')
# plt.legend(loc='best')
# plt.show()
plt.loglog(error,label='Newton')
plt.loglog(error2,label='Pythag')
plt.loglog(error3,label='Basel')
plt.loglog(error4,label='Darts')
plt.loglog(error5,label='Pins')
plt.legend(loc='best')
plt.show()
