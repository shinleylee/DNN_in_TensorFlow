# https://mp.weixin.qq.com/s/pb8MOFZ5iLNswMew9HOxkA
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sb
from scipy.io import loadmat


def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)  # return k numbers range between 0-m

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids


def find_closest_centroids(X, centroids):
    m = X.shape[0]  # how many dots
    k = centroids.shape[0]  # how many classes/centroids
    idx = np.zeros(m)  # record the class every dot belongs to, initialize to zero at first

    for i in range(m):
        min_dist = 1000000  # condition to stop iteration
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                # record the min distance and the index of centroid
                min_dist = dist
                idx[i] = j
    return idx


def compute_centroids(X, idx, k):
    m, n = X.shape  # m=300, n=2
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
        # take the average as the new centroid
    return centroids


def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids


# load data
data = loadmat('dataforK-Means\ex7data2.mat')
X = data['X']
# X.shape[0]=how many dots;
# X.shape[1]=the dimension of every dot (2D or 3D...)
initial_centroids = init_centroids(X, 3)
# initial_centroids = np.array([[3,3],[6,2],[8,5]])

# initially classify every dot into classes/centroids first time
# idx = find_closest_centroids(X, initial_centroids) # get an array 'idx' shows which class every dot belongs to

# compute new centroids of every class
# compute_centroids(X, idx, 3)

# run k-means
idx, centroids = run_k_means(X, initial_centroids, 10)
print(centroids, '\n', idx)
cluster1 = X[np.where(idx == 0)[0], :]  # get all the dots belongs to class 1
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

# draw the figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()
