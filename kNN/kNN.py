import numpy as np
import operator

'''
kNN: k Nearest Neighbors

@author: Wentao Kuang
'''


'''
kNN: k Nearest Neighbors Algorithm

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label
'''
def kNNClisifier(inX, dataSet, lables, k):

    #caculating the distance
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndices = distances.argsort()

    #Choose k nearest neighbours
    classCount={}
    for i in range(k):
        voteIlable = lables[sortedDistIndices[i]]
        classCount[voteIlable] = classCount.get(voteIlable,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''
Read txt file and convert to matrix

Input:      filename: file address
            Obs: numbers of observations

Output:     returnMat: converted observations matrix
            lables: converted class lables vector
            
'''
def file2matrix(filename, Obs):

    #read file
    f = open(filename)
    lines = f.readlines()
    numberOfLines = len(lines)

    #create return matrix
    returnMat = np.zeros((numberOfLines, Obs))
    lables = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:Obs]
        lables.append(listFromLine[-1])
        index += 1
    return returnMat, lables


'''
Feature normalization 

newValue = (oldValue - min) / (max - min)

Input:      dataSet: original dataset

Output:     normDataSet: converted Normalized dataset
            ranges: data ranges
            minVal: minimal data
'''
def norm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals





