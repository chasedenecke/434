import sys
import numpy as np
import random

class KMeans:
    def __init__(self):
        self.trialsPerK = 10
        self.k = int(sys.argv[1])
        digitsFile = open("p4-data.txt", "r")
        self.digits = np.genfromtxt(digitsFile, dtype=np.uint8, delimiter=",")
        self.randomSeeds = []
        self.selectRandomSeeds()
        print("self.randomSeeds = ", self.randomSeeds)
        self.clusters = [Cluster(self.digits[self.randomSeeds[x]]) for x in range(0, self.k)]
        print("self.clusters[0].center = ", self.clusters[0].center)

    def selectRandomSeeds(self):
        self.randomSeeds = random.sample(range(0, self.digits.shape[0]+1), self.k)

    def initializeClusters(self):

        # Assign every point to a cluster
        for i, x in enumerate(self.digits):
            xIsSeed = False

            # Loop through the list of random seeds to see if current point is already part of a cluster (we don't want a cluster to contain duplicates of a point)
            for y in self.randomSeeds:
                if i == y:
                    xIsSeed = True
            if(xIsSeed == False):

                # Find the cluster point x is closest to and append it to that cluster
                closestClusterIndex = 0
                smallestDistance = np.linalg.norm(self.clusters[0] - x) # Compute the distance between the point and the first cluster
                for j, y in enumerate(self.clusters[1:]):

                    # Compare the distance to the new cluster with the smallest distance found so far.
                    # If new cluster is closer to point, update smallest distance and index of cluster with smallest distance.
                    if np.linalg.norm(y - x) < smallestDistance: 
                        closestClusterIndex = j
                        smallestDistance = np.linalg.norm(y - x)
                self.clusters[closestClusterIndex].points.append(x)
        # Update the center points of the clusters
    
    def updateCenters(self):
        for x in self.clusters:
            x.center = np.mean(x.points, axis=0)
    
    def mse(self):
        for x in clusters:
            for y in x.points:
                

    def optimizeClusters(self):
        self.initializeClusters()
        self.updateCenters()
        previousMSE = self.mse()
        # while(cluster has changed from last time):
            # updateClusters()
            # updateCenters()
        
                    

class Cluster:
    def __init__(self, center):
        self.center = center
        self.points = center
Cluster = KMeans()