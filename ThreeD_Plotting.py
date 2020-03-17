import matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

w1 = 1
w2 = -2
w3 = 2
w4 = 1
w5 = 1
w6 = 1

def f(x, y,w1,w2):
    # print(1/(1+np.exp(-w1*x)*np.exp(w2*y)))
    return 1/(1+np.exp(-w1*x)*np.exp(-w2*y))

x = np.linspace(-6, 6, 300)
y = np.linspace(-6, 6, 300)

X, Y = np.meshgrid(x, y)
# print("X",X.shape)
# print("Y",Y.shape)
Z1 = f(X, Y,w1,w2)
Z2 = f(X, Y,w4,w3)
# print("Z",Z.shape)
# Z1,Z2 = np.meshgrid(Z,Z)
Zf = f(Z1,Z2,w5,w6)

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z1, 50, cmap='binary')
ax.contour3D(X, Y, Zf, 50, cmap='plasma')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()