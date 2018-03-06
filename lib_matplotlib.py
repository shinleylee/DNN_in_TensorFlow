import matplotlib.pyplot as plt
import numpy as np

# set the figure size(feet)
plt.figure(figsize=(6, 5), dpi=120)  # figsize=(wide,high)

# assign the x and y
x = np.linspace(-np.pi, np.pi, 256)  # x axis min, max, sample density(no need to change); np.pi=3.1415926...
y1 = np.sin(x)
y2 = x**2

# design the axis
plt.xticks([-np.pi, 0, np.pi], ['-π', '0', 'π'])  # show the point in first [] with corresponding character in second []
plt.yticks([0, 2, 4, 6, 8, 10])  # denote which points are shown on the axis
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# add a figure title
plt.title("Hello world, matplotlib!")

# draw multiple functions and change line style
plt.plot(x, y1, 'b--')
plt.plot(x, y2, 'r-.')
plt.show()
