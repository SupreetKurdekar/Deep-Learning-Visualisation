import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import visual_functions as vf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider
import visual_functions as vf



x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)


X_ = np.expand_dims(X,axis=2)
Y_ = np.expand_dims(Y,axis=2)
data = np.concatenate((X_,Y_),axis=2)
data = data.reshape(900,2)
# dataTensor = torch.from_numpy(data)
dataTensor = torch.Tensor(data)
# print(dataTensor.type)

truth = vf.paraboloid(X, Y)/70
truth_ = truth.reshape(900, 1)
truth_ = torch.tensor(truth_)

fig = plt.figure(figsize=plt.figaspect(0.5))
# first subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
# model parts
model = vf.Perceptron()
criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
optimiser = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, 
                eps=1e-08, weight_decay=0, momentum=0, centered=False)

def on_move(event):
    if event.inaxes == ax:
        ax1.view_init(elev=ax.elev, azim=ax.azim)
    elif event.inaxes == ax1:
        ax.view_init(elev=ax1.elev, azim=ax1.azim)
    else:
        return
    fig.canvas.draw_idle()

num_epochs = 1000
for epochs in range(num_epochs):

    optimiser.zero_grad()

    outPut = model(dataTensor)
    # print("output shapr",outPut.shape)

    loss = criterion(outPut.float(), truth_.float())
    
    loss.backward()
    
    optimiser.step()
    out = outPut.float().detach().numpy().reshape(30, 30)
    # print(out)
    # update(out, X, Y, ax)
    ax.clear()
    ax1.clear()
    plt.ion()
    ax.plot_surface(X, Y, out, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, truth, rstride=1, cstride=1,
                    cmap='Reds', edgecolor='none', alpha=0.3)


    error = out - truth

    # plot of error
    ax1.plot_surface(X, Y, error, rstride=1, cstride=1,
                     cmap='Reds', edgecolor='none', alpha=0.75)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    plt.show()
    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    # time.sleep(0.1)

    # ani = animation.FuncAnimation(fig, update, interval=1000)


    print("loss",loss)

