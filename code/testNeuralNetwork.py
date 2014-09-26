
import sys
import numpy as np
from pybrain.tools.xml.networkreader import NetworkReader

def checkPerformance(nn,tstIn,tstOut,numClasses,threshold):   
  correct = 0; wrong = 0; ctr = 0; counts = 0
  foundQuasars=0; foundWD=0; foundStars=0
  Quasars=0; WD=0; Stars=0
  QuasarsTh=0; WDTh=0; StarsTh=0  
  W = np.mat(np.zeros((3,3)))
  
  for row in range(0,len(tstOut)):
    tempListOut = []; tempListIn = []
    tempListOut.append(int(tstOut[row]))
    for i in range(0,5): tempListIn.append(tstIn[row][i])

    res = nn.activate(tempListIn)
    maxpos = 0; max = res[0]
    for k in range(0,numClasses):
      if res[k] > max:
        max = res[k]
        maxpos = k
    ctr += 1    
    key = int(tstOut[row])    
    if int(tstOut[row]) == 0: Quasars += 1
    if int(tstOut[row]) == 1: WD += 1
    if int(tstOut[row]) == 2: Stars += 1

    if max < threshold:    
      if int(tstOut[row]) == 0: QuasarsTh += 1
      if int(tstOut[row]) == 1: WDTh += 1
      if int(tstOut[row]) == 2: StarsTh += 1
    
    if max > threshold:    
      counts += 1 
      if int(tstOut[row]) == maxpos: correct += 1
      else: wrong += 1 
            
      if maxpos == 0:  foundQuasars += 1
      if maxpos == 1:  foundWD += 1       
      if maxpos == 2:  foundStars += 1
      
      if maxpos == 0 and key == 0: W[0,0] += 1
      if maxpos == 0 and key == 1: W[0,1] += 1
      if maxpos == 0 and key == 2: W[0,2] += 1
      if maxpos == 1 and key == 0: W[1,0] += 1
      if maxpos == 1 and key == 1: W[1,1] += 1
      if maxpos == 1 and key == 2: W[1,2] += 1
      if maxpos == 2 and key == 0: W[2,0] += 1
      if maxpos == 2 and key == 1: W[2,1] += 1
      if maxpos == 2 and key == 2: W[2,2] += 1
      
  return float(foundQuasars)/Quasars,float(foundWD)/WD,float(foundStars)/Stars,Quasars,WD,Stars,QuasarsTh,WDTh,StarsTh,float(counts)/ctr,correct,wrong,float(correct)/float(correct+wrong),int(W[0,0]),int(W[0,1]),int(W[0,2]),int(W[1,0]),int(W[1,1]),int(W[1,2]),int(W[2,0]),int(W[2,1]),int(W[2,2])  

def checkPerformanceTestSet(tstFileName,numF,numC,minVals,maxVals,nnFile,threshold):
  
  data = np.genfromtxt(tstFileName)
  tstIn = data[:,0:5]
  tstOut   = data[:,6]
  tstOut = [int(val) for val in tstOut]

  for i in range(0,len(tstIn)):
    for j in range(0,numF):
      tstIn[i,j] = (tstIn[i,j]-minVals[j])/(maxVals[j]-minVals[j])

  myNetwork = NetworkReader.readFrom(nnFile)  
  return checkPerformance(myNetwork,tstIn,tstOut,numC,threshold)    

def main():

  numClasses = 3
  numFeatures = 5  
  tstFile = "data/testSet.dat"
  outFile = "output/outputFile.dat"
  nnFile = "neuralNetWeights/nn8h.xml"
  
  # These were computed from the *training set*
  minVals = [14.132, 13.795, 13.467, 13.272, 12.848]
  maxVals = [26.95818, 26.07362, 26.58421, 24.36193, 23.64385]

#  threshold = 0.9
#  print checkPerformanceTestSet(tstFile,numFeatures,numClasses,minVals,maxVals,nnFile,threshold)
#  sys.exit()
    
  f = open(outFile, "w")
  for threshold in [0.6+(i/100.) for i in range(0,40,1)]:
    cc = checkPerformanceTestSet(tstFile,numFeatures,numClasses,minVals,maxVals,nnFile,threshold)
    f.write(str(threshold) + " ")
    for i in range(0,22): f.write(str(cc[i]) + " ")
    f.write("\n")  
  f.close()  
      
# 1 Threshold
# 2 foundQuasars/Quasars
# 3 foundWD/WD
# 4 foundStars/Stars
# 5 Number of Quasars in test set
# 6 Number of WD in test set
# 7 Number of Stars in test set
# 8 Quasars outside Threshold
# 9 WD outside Threshold
# 10 Stars outside Threshold
# 11 fraction of data that remains after thresholding.
# 12 Number of correct id's
# 13 Number of wrong id's
# 14 Fraction of correct id's correct/(correct+wrong)
# 15 Number of quasars identified as quasars
# 16 Number of WD identified as quasars
# 17 Number of Stars identified as quasars
# 18 Number of quasars identified as WD
# 19 Number of WD identified as WD
# 20 Number of Stars identified as WD
# 21 Number of quasars identified as Stars
# 22 Number of WD identified as Stars
# 23 Number of Stars identified as Stars

if __name__ == "__main__":
  main()
  