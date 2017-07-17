'''
NaiveBayes Algorithm

@author: Wentao Kuang
'''
from numpy import *

'''
Find all the different vocabularies

Input:      dataset

Output:     vocaset: a set of vocabularies
'''
def createVocabList(dataset):
    vocabSet= set([])
    for document in dataset:
        vocabSet = vocabSet | set(document) # union sets
    return list(vocabSet)

'''
check the existence of vocab from vocablist in the document
based on the set of words model

Input:      vocablist
            inputSet

Output:     returnVec: a set of binary value about the existence
'''
def checkExistence(vocabList, inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print "the word: %s is not in my vocabList!"%word
    return returnVec

'''
check the number of existence of vocab from vocablist in the document
based on the bag of words model

Input:      vocablist
            inputSet

Output:     returnVec: a set of binary value about the existence
'''
def checkExistenceNum(vocabList, inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print "the word: %s is not in my vocabList!"%word
    return returnVec

'''
calculate the probabilities for training set

Input:      trainMatrix: existence vocablist
            trainCategory: corresponding labels

Output:     p0Vec: the probability of each word appeared for label 0
            p1Vec: the probability of each word appeared for label 1
            p1: the probability of label 1
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    p1=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Demon=2.0
    p1Demon=2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Demon += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Demon += sum(trainMatrix[i])
    p1Vec = log(p1Num/p1Demon)
    p0Vec = log(p0Num/p0Demon)
    return p0Vec,p1Vec,p1

'''
NaiveBayes Algorithms

p(c|w)=p(w|c)p(c)/p(w)

Input:      vec2Classify: vector to classify
            p0Vec: the probability of each word appeared for label 0
            p1Vec: the probability of each word appeared for label 1
            p1: the probability of label 1

Output:     binary value
'''

def classifyNB(vec2Classify, p0Vec, p1Vec, p1):
    p1 = sum(vec2Classify*p1Vec) + log(p1)
    p0 = sum(vec2Classify*p0Vec) + log(1-p1)
    if p1>p0:
        return 1
    else:
        return 0






