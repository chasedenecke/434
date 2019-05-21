# Using Google's style gudeline for indentation.

import sys
import numpy as np
import random
import math

MAX_ITERATIONS = 100

# Class for loading the dataset.
class LoadData:
  def __init__(self, FILE):
    digitsFile = open(FILE, "r")
    self.digits = np.genfromtxt(digitsFile, dtype=np.uint8, delimiter=",")

  def featureCount(self):
    return np.size(self.digits, 0)

  def getData(self):
    return self.digits

# Starting KMEANS stuff.
def stoped(oldCentroids, centroids, iterations):
  ## DEAPTH LIMITER
  if iterations > MAX_ITERATIONS:
    return True
  ## Stoping case, return true if old = new.
  return oldCentroids == centroids

def EuclidDist(p1, p2):
  return math.sqrt(np.sum(np.pow((p1 - p2), 2)))

def closestCentroid(x, centroid):
  size = sys.maxint
  c = 0

  for i, cen in enumerate(centroid):
    dist = EuclidDist(x, cen)
    if dist < size:
      size = dist
      c = i

  return c
    
# Classify each piece of data in the dataset
def classify(dataset, centroids):
  labels = []
  # Make tuples of data and centoidID it belongs to.
  for i, x in enumerate(dataset):
    c = closestCentroid(x, centroids)
    labels.append((i, c))

def genRandomCentroid(featureCount, k):
  randomSeeds = []
  randomSeeds = random.sample(range(0, featureCount), k)
  return randomSeeds

def kmeans(dataSet, k):
  featureCount = dataSet.featureCount()
  centroids = genRandomCentroid(featureCount, k)
  
  iterations = 0
  oldCentroids = None

  while not stoped(oldCentroids, centroids, iterations):
    oldCentroids = centroids
    iterations += 1
   
    labels = classify(dataSet, centroids)
    # TODO: Genereat new centroids
      #centroids = getCentroids(dataSet, labels, k, SSE_stack)

  return centroids

if __name__ == "__main__":
  FILE = LoadData("debug-data.txt")
  var = FILE.getData()
  kmeans(FILE, int(sys.argv[1]))
  
