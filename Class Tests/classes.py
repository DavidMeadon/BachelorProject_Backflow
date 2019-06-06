
class Tester:
    def __init__(self,imp1 = 0,imp2 = 0):
        self.num1 = imp1
        self.num2 = imp2

    def sumData(self):
        print("{0} + {1} = {2}".format(self.num1,self.num2,self.num1+self.num2))

class Fibonacci:
    num = 0
    def __init__(self,N = 0):
        if N == 0:
            print(0)
        fibvals = []
        fibvals.append(1)
        fibvals.append(1)
        for i in range(N):
            print(fibvals[i%2])
            fibvals[i%2] = fibvals[i%2] + fibvals[(i+1)%2]
        self.num = fibvals[-1]

class LinRegression:
    alpha = 0
    beta = 0
    def __init__(self, x, y):
        if len(x) != len(y):
            print("Size of vectors to fit must be the same")
            return
        xmean = sum(x)/len(x)
        ymean = sum(y)/len(y)
        num = 0
        denom = 0
        for i in range(len(x)):
            num += (x[i] - xmean)*(y[i] - ymean)
            denom += (x[i] - xmean)**2
        self.alpha = num/denom
        self.beta = ymean - self.alpha*xmean

    def expression(self):
        print("The linear fit for the data is: y = {0}x + {1}".format(self.alpha, self.beta))

