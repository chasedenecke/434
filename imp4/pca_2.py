import numpy as np
import matplotlib.pyplot as plt


class LoadData:
  def __init__(self, FILE):
    File = open(FILE, "r")
    self.data = np.genfromtxt(File, delimiter=",")

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
    self.cov = np.cov(self.data.T)
    return self.cov

  def getEiganValues(self, p=False):
    eigVal, eigVec = np.linalg.eigh(self.cov)

    if p == True:
      for i in range(len(eigVal)):
        eigCov = eigVec[:, i].reshape(1, len(eigVal)).T
      
        print("Eiganvector {}: \n:{}".format(i+1, eigVec[i]))
        print("Eiganvalue {}: \n:{}".format(i+1, eigVal[i]))
        print(20 * '#')

    return eigVal, eigVec

  def getTopEigenValues(self, eigVal, eigVec, n=10):
    eigMap = [(np.abs(eigVal[i]), eigVec[:, i]) for i in range(len(eigVal))]
    eigMap.sort(key=lambda x: x[0], reverse=True)
    return np.asarray([eigMap[i][0] for i in range(n)]), np.asarray([eigMap[i][1] for i in range(n)])

def makeGraphs(featureMean, eigVec, rep):
  fig = plt.figure()
  fig.subplots_adjust(hspace=0.6, wspace=0.6)
  ax = fig.add_subplot(3, 4, 1)
  ax.set_title('Mean image')
  ax.imshow(featureMean.reshape((28, 28)))

  for i, v in enumerate(eigVec):
    ax = fig.add_subplot(3, 4, i + 2)
    name = "eigen-vec " + str(i + 1)
    ax.set_title(name)
    ax.imshow(v.reshape((28, 28)))
  
  plt.show()
  

if __name__ == "__main__":
  FILE = LoadData("p4-data.txt")

  pca = PCA(FILE)
  pca.setFeatureMean()

  pca.getCovMat()
  eigVal, eigVec  = pca.getEiganValues()
  topTenVal, topTenVec = pca.getTopEigenValues(eigVal, eigVec)

  newRep = np.matmul(topTenVec, pca.data.T).T
  
  makeGraphs(pca.featureMean, topTenVec, newRep)

