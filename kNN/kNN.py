from numpy import *
import operator

#create dataset
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group, lables


#kNN Algorithm implementation
def kNNClisifier(inX, dataSet, labels, k):

    #caculating the distance
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndices = distances.argsort()

    #Choose k nearest neighbours
    classCount={}
    for i in range(k):
        voteIlable = lables[sortedDistIndices[i]]
        classCount[voteIlable] = classCount.get(voteIlable,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


#test kNN
group,lables=createDataSet()
print(kNNClisifier([1,1],group,lables,3))


