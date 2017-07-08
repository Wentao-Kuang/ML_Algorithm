from math import log
import operator
'''
Decision Tree (DT)

@author: Wentao Kuang
'''


'''
calculate Shannon Entropy = -sum_{1,n} p(x_{i})*log_{2}*p_{i}

Input:      dataSet: Matrix data set
        
Output:     Shannon Entropy
'''
def ShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    #count all numbers of each class
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #calculate entropy for each class
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt



'''
Split Dataset 

Input:      dataSet: dataSet to be splited
            axis: feature to be scaned
            value: feature value need to selected

Output:     splited dataset
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #get all other values in feaVec
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
Split Dataset with best feature use shannon entropy

Input:      dataSet: dataSet to be splited
            
Output:     best feature
'''
def chooseBestFeature(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = ShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        #get all the values from feature i
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * ShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


'''
Sort list return the most frequency value.

'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=true)
    return sortedClassCount[0][0]

'''
Build decision Tree
'''
def createTree(dataSet, lables):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0] #All the lables are same
    if len(dataSet[0]) == 1:
        return majorityCnt(classList) #No feature left
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = lables[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(lables[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = lables[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)