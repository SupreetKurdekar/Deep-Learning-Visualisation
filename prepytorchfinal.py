import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider
import visual_functions as vf

# fig, ax = plt.subplots()
w1 = 3
w2 = 1
w3 = 1
w4 = 1
w5 = 1
w6 =1

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)

Z1 = vf.f(X, Y, w1, w2)
Z2 = vf.f(X, Y, w3, w4)
Zf = vf.f(Z1, Z2, w5, w6)

truth = vf.paraboloid(X,Y)/70

error = Zf - truth

fig = plt.figure(figsize=plt.figaspect(0.5))

# first subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.plot_surface(X, Y, Zf, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, truth, rstride=1, cstride=1,
                cmap='Reds', edgecolor='none',alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax1 = fig.add_subplot(1, 2, 2, projection='3d')

# plot of error
ax1.plot_surface(X, Y, error, rstride=1, cstride=1,
                cmap='Reds', edgecolor='none',alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

axamp1 = plt.axes([0.25, 0.03, 0.50, 0.02])
axamp2 = plt.axes([0.25, 0.06, 0.50, 0.02])
axamp3 = plt.axes([0.25, 0.09, 0.50, 0.02])
axamp4 = plt.axes([0.25, 0.12, 0.50, 0.02])
axamp5 = plt.axes([0.25, 0.15, 0.50, 0.02])
axamp6 = plt.axes([0.25, 0.18, 0.50, 0.02])
# Slider
samp1 = Slider(axamp1, 'W1', -10, 10, valinit=w1)
samp2 = Slider(axamp2, 'W2', -10, 10, valinit=w2)
samp3 = Slider(axamp3, 'W3', -10, 10, valinit=w3)
samp4 = Slider(axamp4, 'W4', -10, 10, valinit=w4)
samp5 = Slider(axamp5, 'W5', -10, 10, valinit=w5)
samp6 = Slider(axamp6, 'W6', -10, 10, valinit=w6)

def update(val):
    # amp is the current value of the slider
    w1 = samp1.val
    w2 = samp2.val
    w3 = samp3.val
    w4 = samp4.val
    w5 = samp5.val
    w6 = samp6.val
    # update curve
    Z1 = vf.f(X, Y, w1, w2)
    Z2 = vf.f(X, Y, w3, w4)
    Zf = vf.f(Z1, Z2, w5, w6)
    ax.clear()
    ax.plot_surface(X, Y, truth, rstride=1, cstride=1,
                    cmap='Blues', edgecolor='none', alpha=0.1)
    ax.plot_surface(X, Y, Zf, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    error = Zf - truth
    ax1.clear()
    ax1.plot_surface(X, Y, error, rstride=1, cstride=1,
                     cmap='Reds', edgecolor='none', alpha=0.5)
    # redraw canvas while idle
    fig.canvas.draw_idle()

def on_move(event):
    if event.inaxes == ax:
        ax1.view_init(elev=ax.elev, azim=ax.azim)
    elif event.inaxes == ax1:
        ax.view_init(elev=ax1.elev, azim=ax1.azim)
    else:
        return
    fig.canvas.draw_idle()

c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

# call update function on slider value change
samp2.on_changed(update)
samp1.on_changed(update)
samp3.on_changed(update)
samp4.on_changed(update)
samp5.on_changed(update)
samp6.on_changed(update)
plt.show()