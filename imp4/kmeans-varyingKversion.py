import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import code

class KMeans:
    def __init__(self, k):
        self.trialsPerK = 10
        self.k = k
        digitsFile = open("p4-data.txt", "r")
        print("Loading data...")
        # Read in the digits from the data file and store them in a numpy array.
        # Since values are 0-255, use np.uint8 for data type to minize storage space required.
        self.digits = np.genfromtxt(digitsFile, dtype=np.uint8, delimiter=",")
        self.SSErrors = [sys.maxsize]
        self.randomSeeds = None
        self.clusters = None
        self.selectRandomSeeds()
        self.clusters = [Cluster(self.digits[self.randomSeeds[x]]) for x in range(0, self.k)]
        # This complicated line produces a list of clusters whose size is equal to K.
        # The center of each cluster and the list of points in the cluster are initialized to
        # the value of the random seed point chosen by selectRandomSeeds()

    def selectRandomSeeds(self):
        self.randomSeeds = random.sample(range(0, self.digits.shape[0]), self.k)

    def reassignRandomSeeds(self):
        self.randomSeeds = random.sample(range(0, self.digits.shape[0]), self.k)
        for i, x in enumerate(self.clusters):
            x.center = self.digits[self.randomSeeds[i]]
            x.points = [self.digits[self.randomSeeds[i]]]

    def initializeClusters(self):
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
                self.clusters[closestClusterIndex].points.append(x)    
    # Checks all points to see if the cluster they belong to needs to be switched.
    # Does so by asprint(type(x[0]))print(type(x[0]))signing all points to the cluster whose center is closest using euclidean distance.
    def updateClusters(self):

        
        for i, x in enumerate(self.clusters):
            tempList = []

            # Loops once for each point in the dataset
            for j, y in enumerate(x.points): 
                betterCluster = None

                # Check the point against each cluster
                for k, z in enumerate(self.clusters):
                    # print("k = ", k)
                    if np.linalg.norm(z.center - y) < np.linalg.norm(x.center - y):
                        betterCluster = k
                
                # If a better cluster for the point has been found, 
                # append the point to that cluster
                if betterCluster != None:
                    self.clusters[betterCluster].points.append(y)
                
                # If there is no better cluster, append the point to a temporary array that the current
                # cluster will be set equal to after every point in the cluster is checked.
                # This is done instead of deleting the point from the cluster to avoid problems with
                # out of bounds indexing that occurs if one tries to delete the rows within the for loop
                else:
                    tempList.append(x.points[j])
            x.points = tempList
    
    def updateCenters(self):
        for x in self.clusters:
            x.center = np.mean(x.points, axis=0)
    
    def sse(self):
        sse = 0
        for x in self.clusters:
            for y in x.points:
                sse += np.linalg.norm(y - x.center)**2
        self.SSErrors.append(sse)
        if len(self.SSErrors) != 1:
            print(self.SSErrors[-1])

    def optimizeClusters(self):
        self.initializeClusters()
        self.updateCenters()
        self.sse()
        counter = 0
        while True:

            counter += 1
            self.updateClusters()
            self.updateCenters()
            self.sse()
            if self.SSErrors[-1] == self.SSErrors[-2]:
                break
    
    def displayResults(self):
        plt.plot(range(len(self.SSErrors)), self.SSErrors)
        # plt.ylim(10000000000, 20000000000)
        plt.show()
    
    def getFinalSSE(self):
        return self.SSErrors[-1]
                    

class Cluster:
    def __init__(self, center):
        self.center = center
        self.points = [center]

if __name__ == "__main__":
    
    smallestSSE = []
    for x in range(2, 12):
        cloud = KMeans(x)
        tempSmallest = sys.maxsize
        print("k value: ", x)
        for y in range(0, 10):   
            print("run ", y)
            cloud.optimizeClusters()
            currentSSE = cloud.getFinalSSE()
            if currentSSE < tempSmallest:
                tempSmallest = currentSSE
            cloud.reassignRandomSeeds()
        smallestSSE.append(tempSmallest)
        print("smallestSSE = ", smallestSSE)

    print(smallestSSE)
    plt.plot([i for i in range(2, len(smallestSSE)+2)], smallestSSE)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Converged Sum Squared Error')
    plt.title('Sum Squared Error')
    plt.show()
    code.interact(local=locals()) # Start interpreter for debugging
