'''
Documentation classification Example
Identify abusive document.
@author: Wentao Kuang
'''
from numpy import *
from bayes import *


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


'''
testing NaiveBayes Algorithms

'''


def testingNB():
    dataset,labels=loadDataSet()
    myVocbList=createVocabList(dataset)
    trainMatrix=[]
    for doc in dataset:
        trainMatrix.append(checkExistence(myVocbList,doc))
    p0v,p1v,p1=trainNB0(trainMatrix,labels)
    testEntry=['love','my','dalmation']
    testMatrix = array(checkExistence(myVocbList,testEntry))
    print testEntry,' classified as: ',classifyNB(testMatrix,p0v,p1v,p1)
    testEntry=['stupid','garbage']
    testMatrix=array(checkExistence(myVocbList,testEntry))
    print testEntry,' classified as: ',classifyNB(testMatrix,p0v,p1v,p1)

