import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def f1(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def f(x, y,w1,w2):
    # print(1/(1+np.exp(-w1*x)*np.exp(w2*y)))
    return 1/(1+np.exp(-w1*x)*np.exp(-w2*y))

def paraboloid(x,y):
    return x**2 + y**2



class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,1)
        self.sigmoid = torch.nn.Sigmoid() # instead of Heaviside step fn
    
    def forward(self, x):
        output = self.fc1(x)
        # output = self.sigmoid(output) # instead of Heaviside step fn
        output = self.fc2(output)
        output = self.sigmoid(output)

        return output