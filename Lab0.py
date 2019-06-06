import numpy as np
import matplotlib.pyplot as plt
import testfuncs as tf

    
tf.fibonacci1(8)


N = 8
for i in range(N):
    print(tf.fibonacci2(i+1))



print(abs(tf.heron(np.pi,1) - np.sqrt(np.pi)))

# Linear Regression

N = int(1e6)
x = []
y = []
for idx in range(N+1):
    x.append(idx/N)
    y.append(x[idx] + np.random.standard_normal(1))

alpha,beta = tf.listRegression(x,y)

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


alpha2,beta2 = tf.numpRegression(x2,y2)

# Using iPython more than 100 times faster

