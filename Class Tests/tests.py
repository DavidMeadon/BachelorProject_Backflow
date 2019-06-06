from classes import *
import numpy as np

newtest = Tester(4,4)

newtest.sumData()

# test = Fibonacci(8)

# print(test.num)
N = int(1e3)
x2 = np.zeros((N,1))
y2 = np.zeros((N,1))

for idx in range(N):
    x2[idx] = idx/N
    y2[idx] = x2[idx] + np.random.standard_normal(1)

LinRegression(x2,y2).expression()

# fit.expression()