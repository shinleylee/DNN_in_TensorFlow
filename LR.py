# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# generate the data
x = [1, 2, 3, 4, 6, 7, 8, 9, 10]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1]
train_x = np.asarray(np.row_stack((np.ones(shape=(1, len(x))), x)), dtype=np.float64)
train_y = np.asarray(y, dtype=np.float64)
train_W = np.asarray([-1, -1], dtype=np.float64).reshape(1, 2)


# define h_theta(x)
def sigmoid(X):
    return 1 / (1 + np.power(np.e, -X))


# define J(theta)
def lossfunc(X, Y, W):
    n = len(Y)
    return (-1 / n) * np.sum(Y * np.log(sigmoid(np.matmul(W, X))) + (1 - Y) * np.log((1 - sigmoid(np.matmul(W, X)))))


# define gradient descent
loss_Trace = []
w_Trace = []
b_Trace = []


def gradientDescent(X, Y, W, lr=0.001, iters=500):
    n = len(Y)
    for i in range(iters):
        W = W - (lr / n) * np.sum((sigmoid(np.matmul(W, X)) - Y) * X, axis=1)
        # for record in gif
        if (i < 100 and i % 2 == 0) or (i % 1000 ==0):
            b_Trace.append(W[0, 0])
            w_Trace.append(W[0, 1])
            loss_Trace.append(lossfunc(X, Y, W))
    return W


final_W = gradientDescent(train_x, train_y, train_W, lr=0.3, iters=100000)


# output and draw the graph
print("Final Weight:", final_W)
print("Weight details trace", np.asarray(([b_Trace, w_Trace])))
print("Loss details trace", loss_Trace)

fig, ax = plt.subplots()
ax.scatter(np.asarray(x), np.asarray(y))
ax.set_title(r'$Fitting\ line$')


def update(i):
    try:
        ax.lines.pop(0)
    except Exception:
        pass
    plot_X = np.linspace(-1, 12, 100)
    W = np.asarray([b_Trace[i], w_Trace[i]]).reshape(1, 2)
    X = np.row_stack((np.ones(shape=(1, len(plot_X))), plot_X))
    plot_Y = sigmoid(np.matmul(W, X))
    line = ax.plot(plot_X, plot_Y[0], 'r-', lw=1)
    ax.set_xlabel(r"$Cost\ %.6s$" % loss_Trace[i])
    return line


ani = animation.FuncAnimation(fig, update, frames=len(w_Trace), interval=100)
ani.save('logisticregression.html', writer='imagemagick')

plt.show()
