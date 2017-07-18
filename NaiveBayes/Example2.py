'''
Spam email classification Example
@author: Wentao Kuang
'''

import re
from bayes import *
'''
Split email to words

Input:      email file location

Output:     tokens: a set of words from email
'''
def splitEmail(file):
    emailText = open(file).read()
    regEx = re.compile('\\W*')
    tokens = regEx.split(emailText)
    tokens = [token.lower() for token in tokens if len(token) > 0]
    return tokens


'''
traning email dataset and test error rate

Input:      

Output:    
'''
def spamTest():
    docList=[]
    classList=[]
    fullTest=[]
    # construction data and vocablist
    for i in range(1,26):
        wordList = splitEmail('/Users/kuangwentao/PycharmProjects/ML_Algorithm/DataSets/email/spam/%d.txt'%i)
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList = splitEmail('/Users/kuangwentao/PycharmProjects/ML_Algorithm/DataSets/email/ham/%d.txt'%i)
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocalList = createVocabList(docList)
    # random split traingset and testset
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClass = []
    #training
    for docIndex in trainingSet:
        trainMat.append(checkExistence(vocalList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0,p1,pSpam = trainNB0(array(trainMat),array(trainClass))
    #testing
    errorcount = 0
    for docIndex in testSet:
        wordVector = checkExistence(vocalList,docList[docIndex])
        if classifyNB(array(wordVector),p0,p1,pSpam) != classList[docIndex]:
            errorcount += 1
    print 'the error rate is ', float(errorcount)/len(testSet)




