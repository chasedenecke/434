import numpy as np

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

  def setFeatureMean(self):
    meanStack = []
    for i in range(self.featureCount):
        sigma = np.mean(self.data[i,:])
        meanStack.append(sigma)

    self.featureMean = np.asarray(meanStack)
    return self.featureMean

if __name__ == "__main__":
  #FILE = LoadData("p4-data.txt")
  FILE = LoadData("debug-data.txt")

  pca = PCA(FILE)
  print(pca.setFeatureMean())
