import matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider, Button, RadioButtons
import visual_functions as vf

# w = np.array([1,2,2,1,-1,2]) 
# w = np.array([1,2,-2,6,1,1]) use this weight vector for some fun 
# results
# weight vector
w = np.array([1,1,2,1,-5,2])

# Input point and space generation
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)


Z1 = vf.f(X, Y,w[0],w[1]) # output of neuron 1
Z2 = vf.f(X, Y,w[2],w[3]) # output of neuron 2
Zf = vf.f(Z1,Z2,w[4],w[5]) # final output of nueron 3

truth = vf.paraboloid(X,Y)/70

error = Zf - truth

fig = plt.figure(figsize=plt.figaspect(0.5))

# first subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.plot_surface(X, Y, Zf, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, truth, rstride=1, cstride=1,
                cmap='Blues', edgecolor='none',alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# subplot 2
ax1 = fig.add_subplot(1, 2, 2, projection='3d')

# plot of error
ax1.plot_surface(X, Y, error, rstride=1, cstride=1,
                cmap='Reds', edgecolor='none',alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

plt.show()