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
        self.MSErrors = []
        self.randomSeeds = []
        self.selectRandomSeeds()
        print("self.randomSeeds = ", self.randomSeeds)

        # This complicated line produces a list of clusters whose size is equal to K.
        # The center of each cluster and the list of points in the cluster are initialized to
        # the value of the random seed point chosen by selectRandomSeeds()
        self.clusters = [Cluster(self.digits[self.randomSeeds[x]]) for x in range(0, self.k)]
        print("self.clusters[0].center = ", self.clusters[0].center)

    def selectRandomSeeds(self):
        self.randomSeeds = random.sample(range(0, self.digits.shape[0]), self.k)

    def initializeClusters(self):

        # Assign every point to a cluster
        for i, x in enumerate(self.digits):
            xIsSeed = False

            # Loop through the list of random seeds to see if current point is already part of a cluster (we don't want a cluster to contain duplicates of a point)
            for y in self.randomSeeds:
                if i == y:
                    xIsSeed = True
            if xIsSeed == False:

                # Find the cluster x is closest to and append it to that cluster
                closestClusterIndex = 0
                smallestDistance = np.linalg.norm(self.clusters[0].center - x) # Compute the distance between the point and the first cluster's center
                for j, y in enumerate(self.clusters[1:]):

                    # Compare the distance to the new cluster's center with the smallest distance found so far.
                    # If new cluster is closer to point, update smallest distance and index of cluster with smallest distance.
                    if np.linalg.norm(y.center - x) < smallestDistance: 
                        closestClusterIndex = j
                        smallestDistance = np.linalg.norm(y.center - x)
                tempArray = np.append(self.clusters[closestClusterIndex].points, [x], axis=0)
                self.clusters[closestClusterIndex].points = tempArray
        # Update the center points of the clusters  
    
    def updateClusters(self):
        for x in self.clusters:
            for i, y in enumerate(x.points):
                for z in self.clusters:
                    if np.linalg.norm(y - z.center) < np.linalg.norm(y - x.center):
                        z.points.append(y)
                        x.points = np.delete(x.points, i)

    
    def updateCenters(self):
        for x in self.clusters:
            x.center = np.mean(x.points, axis=0)
    
    def mse(self):
        mse = 0
        for x in clusters:
            for y in x.points:
                mse += np.linalg.norm(y - x.center)
        self.MSErrors.append(mse)

    def optimizeClusters(self):
        self.initializeClusters()
        self.updateCenters()
        previousMSE = self.mse()
        # while(cluster has changed from last time):
            # updateClusters()
            # updateCenters()
        while True:
            self.updateClusters()
            self.updateCenters()
            if self.MSErrors[-1] == self.MSErrors[-2]:
                break
                    

class Cluster:
    def __init__(self, center):
        self.center = center
        self.points = center
Cluster = KMeans()
Cluster.optimizeClusters()