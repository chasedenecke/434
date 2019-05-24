import numpy as np
import matplotlib.pyplot as plt
import code

class LoadData:
  def __init__(self, FILE):
    File = open(FILE, "r")
    self.data = np.genfromtxt(File, dtype=np.uint8, delimiter=",")

  def featureCount(self):
    return np.size(self.data, 1)

  def getData(self):
      return self.data

class PCA:
  def __init__(self, FILE):
    self.data = FILE.getData()
    self.featureCount = FILE.featureCount()
    self.featueMean = None
    self.cov = None

  # Creates a vector of mean values.
  # One for each feature coloumn.
  def setFeatureMean(self):
    meanStack = []
    for i in range(self.featureCount):
         sigma = self.data[:,i].mean()
         meanStack.append(sigma)

    self.featureMean = np.asarray(meanStack)
    return self.featureMean

  def getCovMat(self):
    self.cov = np.corrcoef(self.data)
    return self.cov

  def getEiganValues(self, p=False):
    eigVal, eigVec = np.linalg.eig(self.cov)

    if p == True:
      for i in range(len(eigVal)):
        eigCov = eigVec[:, i].reshape(1, len(eigVal)).T
      
        print("Eiganvector {}: \n:{}".format(i+1, eigVec[i]))
        print("Eiganvalue {}: \n:{}".format(i+1, eigVal[i]))
        print(20 * '#')

    return eigVal, eigVec

  def getTopEiganValues(self, eigVal, eigVec, n=10):
      eigMap = [(eigVal[i], eigVec[i]) for i in range(len(eigVal))]
      eigMap.sort(key=lambda x: x[0], reverse=True)
      return [eigMap[i][0] for i in range(n)], [eigMap[i][1] for i in range(n)]

if __name__ == "__main__":
  FILE = LoadData("debug-data.txt")
  #FILE = LoadData("p4-data.txt")

  pca = PCA(FILE)
  pca.setFeatureMean()
  plt.imshow(np.reshape(pca.featureMean,(28,28)))
  plt.show()
  pca.getCovMat()
  eigVal, eigVec  = pca.getEiganValues()
  topTenVal, topTenVec = pca.getTopEiganValues(eigVal, eigVec)
fig, ax = plt.subplots(nrows=3, ncols=4)
  for i, row in enumerate(ax):
    for j, col in enumerate(row):
      if i == 0 and j == 0:
        plt.imshow(np.reshape(pca.featureMean,(28,28)))
      else:
        plt.imshow(np.reshape(topTenVec[i*4 + j - 1], (28,28)))
  plt.show()
  print(topTenVal)
  code.interact(local=locals()) # Start interpreter for debugging
