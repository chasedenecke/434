class LoadData:
  # Initialize files to be loaded.
  def __init__(self, FILE):
    self.filename = FILE
    self.lables = None # Feature labels if availibe.
    self.data = None # Raw data
    self.longestSequence = None

  # Import raw data from the file.
  def getData(self, readFeatures=True):
    labels = []
    data = []
    with open(self.filename) as fp:
      if readFeatures == True:
        line = fp.readline()
        labels.append(line.split('\t'))
        labels = list(map(str.rstrip, labels[0]))
        self.lables = labels
      else:
        self.lables = None

      for line in fp:
        bucket = line.split('\t')
        data.append(list(map(str.rstrip, bucket)))

    self.data = data
  # Function for sequance data. Makes an nx2 matrix. Coloumn 0 is the
  # feature label. Coloumn 1 is the RNA sequence.
  def mapSequences(self, classNum, n=0):
    data = []

    for i in range(0, len(self.data), 2):
      if len(self.data[i+1][0]) > n:
        n = len(self.data[i+1][0])
      data.append([self.data[i][0].strip('>'), self.data[i+1][0], classNum])

    self.longestSequence = n
    self.data = data
    return n

  def padSequences(self):
    assert(self.longestSequence > 0),"ERROR: Longest sequence is size of 0."
    for seq in self.data:
      dif = self.longestSequence - len(seq[1])
      if (dif > 0):
        seq[1] = seq[1] + str('0' * dif)


if __name__ == "__main__":
  sequences_pkfs = LoadData("pkfs_Train.fasta")
  sequences_pks = LoadData("pks_Train.fasta");

  sequences_pkfs.getData(False)# False because features are not in row 0.
  sequences_pks.getData(False)

  n = sequences_pks.mapSequences(1)
  sequences_pkfs.mapSequences(0, n)

  print("Longest RNA sequenc is: ", n)
  print("Padding Sequences")
  sequences_pkfs.padSequences()
  sequences_pks.padSequences()
  
  with open("sequences.csv", "w") as fp:
    sequences = sequences_pks.data + sequences_pkfs.data
    for row in sequences:
      fp.write(''.join((str(row) + '\n').strip('[]')))
