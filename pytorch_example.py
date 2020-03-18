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
truth = truth.reshape(900, 1)
truth = torch.tensor(truth)

fig = plt.figure(figsize=plt.figaspect(0.5))
# first subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# model parts
model = vf.Perceptron()
criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
optimiser = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, 
                eps=1e-08, weight_decay=0, momentum=0, centered=False)


def update(outPut, X, Y, ax):
    # amp is the current value of the slider
    # update curve
    ax.clear()
    # ax.plot_surface(X, Y, truth, rstride=1, cstride=1,
                    # cmap='Blues', edgecolor='none', alpha=0.1)
    ax.plot_surface(X, Y, outPut, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    # error = Zf - truth
    # ax1.clear()
    # ax1.plot_surface(X, Y, error, rstride=1, cstride=1,
    #                  cmap='Reds', edgecolor='none', alpha=0.5)
    # redraw canvas while idle
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


num_epochs = 10
for epochs in range(num_epochs):

    optimiser.zero_grad()

    outPut = model(dataTensor)
    # print("output shapr",outPut.shape)

    loss = criterion(outPut.float(), truth.float())
    
    loss.backward()
    
    optimiser.step()
    out = outPut.float().detach().numpy().reshape(30, 30)
    print(out.shape)
    update(out, X, Y, ax)
    plt.show()

    print("loss",loss)

