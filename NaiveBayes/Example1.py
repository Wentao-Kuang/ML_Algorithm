'''
Documentation classification Example
Identify abusive document.
@author: Wentao Kuang
'''
from numpy import *


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
