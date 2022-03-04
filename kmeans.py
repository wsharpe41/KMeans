# Create Clusters of points
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()  # for plot styling
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)

# This is another implementation using Numpy Arrays


def createClusters(targets, points):
    targetID = np.zeros((len(points), 1))

    # assign point to target
    for index, point in enumerate(points):
        targetID[index] = getClosest(targets, point)
    return targetID  # this maps points to target

# assign point to closest target


def getClosest(targets, point):
    closest = 0
    dmin = np.infty
    for i in range(len(targets)):
        d = np.linalg.norm(targets[i] - point)
        if d < dmin:
            closest = i
            dmin = d
    return closest


def getRandomPoint(X):
    i = np.random.randint(0, len(X)-1)
    return np.array(X[i])


def updateTargets(X, targetID, targets):
    newTargets = np.zeros((0, 2), float)

    rowsTarget = targets.shape[0]
    rowsX = X.shape[0]
    for i in range(0, rowsTarget):
        points = np.zeros((0, 2), float)
        for j in range(0, rowsX):
            clusterID = int(targetID[j])
            if clusterID == i:
                point = X[j]
                points = np.append(points, point)

        k = int(np.ceil(len(points)/2))
        points = points.reshape((k, 2))
        newTargets = np.append(newTargets, points.mean(0))

    k = int(len(newTargets)/2)
    return newTargets.reshape((k, 2))


# get k random targets
k = 4
targets = np.zeros((k, 2))
for i in range(k):
    centroid = getRandomPoint(X)
    print(centroid)
    targets[i] = np.array(centroid)

print(f'targets AAA: {targets}')
# loop until targets don't move
for iter in range(10):
    targetID = createClusters(targets, X)

    plt.scatter(X[:, 0], X[:, 1], c=targetID, s=50, cmap='viridis')

    # update the targets
    targets = np.array(updateTargets(X, targetID, targets))

    # plot centroids
    print(f'targets: {targets}')
    centx = targets[:, 0]
    centy = targets[:, 1]
    plt.scatter(centx, centy, c='orange')
    plt.xlabel('X')
    plt.ylabel('Y')

centx = targets[:, 0]
centy = targets[:, 1]
plt.scatter(centx, centy, c='red')
plt.xlabel('X')
plt.ylabel('Y')
