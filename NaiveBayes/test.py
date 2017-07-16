from Example1 import *
from bayes import *

dataset,lable=loadDataSet()
vocablist=createVocabList(dataset)
print(vocablist)
print(checkExistence(vocablist,dataset[0]))

trainMatrix =[]
for doc in dataset:
    trainMatrix.append(checkExistence(vocablist,doc))

p0Vec,p1Vec,p0=trainNB0(trainMatrix,lable)
print(p0Vec)
print(p1Vec)
print(p0)

testingNB()