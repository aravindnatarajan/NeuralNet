
import sys
import numpy as np

from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.xml.networkwriter import NetworkWriter

def getMinMax(dataArray, col):
  myMin = dataArray[0,col]; myMax = dataArray[0,col]; myMean = 0.; ctr = 0
  for i in range(0,len(dataArray)):
    if dataArray[i,col] <= 5.: continue
    if dataArray[i,col] > myMax: myMax = dataArray[i,col]
    if dataArray[i,col] < myMin: myMin = dataArray[i,col]  
    myMean += dataArray[i,col]
    ctr += 1
        
  return myMin,myMax,myMean/ctr


def normalizeData(trn,num):
  # Find the min and max of every feature in the training set.
  minArray = []; maxArray = []; meanArray = []
  for feature in range(0,num): 
    minArray.append((getMinMax(trn,feature))[0])
    maxArray.append((getMinMax(trn,feature))[1])
    meanArray.append((getMinMax(trn,feature))[2])    
    
  # Put data in the training set in the range(0,1)
  for i in range(0,len(trn)):
    for j in range(0,num):
      trn[i,j] = (trn[i,j]-minArray[j])/(maxArray[j]-minArray[j])
       
def simpleNeuralNetworkTrain(fileName,numFeatures,numClasses,possibleOutputs,numHiddenNodes,numTrainingEpochs):

  data = np.genfromtxt(fileName)
  trnIn = data[:,0:5]
  trnOut   = data[:,6]
  trnOut = [int(val) for val in trnOut]
  
  normalizeData(trnIn,numFeatures)
  trndata = ClassificationDataSet(numFeatures, possibleOutputs, nb_classes=numClasses)
  for row in range(0,len(trnIn)):
    tempListOut = []; tempListIn = []
    tempListOut.append(int(trnOut[row]))
    for i in range(0,numFeatures): tempListIn.append(trnIn[row][i])
    trndata.addSample(tempListIn,tempListOut)
  
  trndata._convertToOneOfMany()

#  When running for the first time  
  myNetwork = buildNetwork(numFeatures,numHiddenNodes,numClasses,outclass=SoftmaxLayer,bias=True,recurrent=False)

# Read from file after the first try.
#  myNetwork = NetworkReader.readFrom('firstTime.xml')    # Use saved results.
  trainer = BackpropTrainer(myNetwork, dataset=trndata, momentum=0., verbose=True, weightdecay=0.)
  for i in range(numTrainingEpochs): trainer.trainOnDataset(dataset=trndata)
#  NetworkWriter.writeToFile(myNetwork, 'secondTime.xml')  # Record your results!


def main():


  numClasses = 3           # 3 possible output categories: Quasars, White Dwarfs, Stars
  numFeatures = 5          # 5 Training features: Magnitudes: u,g,r,i,z
  numHiddenNodes = 8       # 1 hidden layer, 8 neurons.
  possibleOutputs = 1      # Only 1 output: Either Q,W, or S.
  numTrainingEpochs = 3  # Train 100 times.
  trnFileName = "data/trainingSet.dat"

  simpleNeuralNetworkTrain(trnFileName,numFeatures,numClasses,possibleOutputs,numHiddenNodes,numTrainingEpochs)


if __name__ == "__main__":
  main()
  