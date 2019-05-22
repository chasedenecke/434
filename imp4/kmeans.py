import sys
import numpy as np
import random

class KMeans:
    def __init__(self):
        self.trialsPerK = 10
        self.k = int(sys.argv[1])
        digitsFile = open("debug-data.txt", "r")

        # Read in the digits from the data file and store them in a numpy array.
        # Since values are 0-255, use np.uint8 for data type to minize storage space required.
        self.digits = np.genfromtxt(digitsFile, dtype=np.uint8, delimiter=",")
        self.MSErrors = [sys.maxsize]
        self.randomSeeds = []
        self.selectRandomSeeds()
        print("random seeds: ", self.randomSeeds)
        # This complicated line produces a list of clusters whose size is equal to K.
        # The center of each cluster and the list of points in the cluster are initialized to
        # the value of the random seed point chosen by selectRandomSeeds()
        self.clusters = [Cluster(self.digits[self.randomSeeds[x]]) for x in range(0, self.k)]

    def selectRandomSeeds(self):
        self.randomSeeds = random.sample(range(0, self.digits.shape[0]), self.k)

    def initializeClusters(self):
        print("size of digits = ", self.digits.size)
        print("digits shape: ", self.digits.shape)
        # Assign every point to a cluster
        for i, x in enumerate(self.digits):
            xIsSeed = False

            # Loop through the list of random seeds to see if current point is already part of a cluster (we don't want a cluster to contain duplicates of a point)
            for y in self.randomSeeds:
                if i == y:
                    xIsSeed = True
            if xIsSeed == False:
                closestClusterIndex = None
                smallestDistance = sys.maxsize

                # Find the cluster x is closest to and append it to that cluster
                for j, y in enumerate(self.clusters):

                    # Compare the distance to the new cluster's center with the smallest distance found so far.
                    # If new cluster is closer to point, update smallest distance and index of cluster with smallest distance.
                    if np.linalg.norm(y.center - x) < smallestDistance:
                        closestClusterIndex = j
                        smallestDistance = np.linalg.norm(y.center - x)
                
                # Append the point to the cluster whose center is closest to it
                self.clusters[closestClusterIndex].points = np.append(self.clusters[closestClusterIndex].points, [x], axis=0)
    
    # Checks all points to see if the cluster they belong to needs to be switched.
    # Does so by assigning all points to the cluster whose center is closest using euclidean distance.
    def updateClusters(self):
        # for i, x in enumerate(self.clusters[0].points):
        #     print(i, ": ", type(x))
        for i, x in enumerate(self.clusters):
            # Loops once for each point in the dataset
            for j, y in enumerate(x.points): 
                betterCluster = None

                # Check the point against each cluster
                for k, z in enumerate(self.clusters):
                    # print("k = ", k)
                    print("type of x = ", type(x))
                    print("type of z = ", type(z))
                    if np.linalg.norm(z.center - y) < np.linalg.norm(x.center - y):
                        betterCluster = k
                
                # If a better cluster for the point has been found, 
                # append the point to that cluster and remove it from the old cluster
                if betterCluster != None:
                    print("better cluster found")
                    tempArray = np.append(self.clusters[betterCluster].points, y)
                    self.clusters[betterCluster] = tempArray
                    x.points = np.delete(x.points, j, axis=0)

                    
    
    def updateCenters(self):
        for x in self.clusters:
            x.center = np.mean(x.points, axis=0)
    
    def mse(self):
        mse = 0
        for x in self.clusters:
            for y in x.points:
                mse += np.linalg.norm(y - x.center)
        self.MSErrors.append(mse)
    def optimizeClusters(self):
        self.initializeClusters()
        for x in self.clusters:
            print("num points in cluster = ", x.points.shape)
        self.updateCenters()
        self.mse()
        while True:
            self.updateClusters()
            self.updateCenters()
            self.mse()
            if self.MSErrors[-1] == self.MSErrors[-2]:
                break
                    

class Cluster:
    def __init__(self, center):
        self.center = center
        self.points = np.empty([0, center.shape[0]])
        self.points = np.append(self.points, [center], axis = 0)
Cluster = KMeans()
Cluster.optimizeClusters()