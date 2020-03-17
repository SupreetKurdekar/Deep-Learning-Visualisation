import numpy as np

def f1(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def f(x, y,w1,w2):
    # print(1/(1+np.exp(-w1*x)*np.exp(w2*y)))
    return 1/(1+np.exp(-w1*x)*np.exp(-w2*y))

def paraboloid(x,y):
    return x**2 + y**2