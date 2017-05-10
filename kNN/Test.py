from numpy import *
from kNN import *
'''
kNN: Testing k Nearest Neighbors

@author: Wentao Kuang
'''

#create test dataset
def createTestDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

#test kNN
group, lables=createTestDataSet()
print(kNNClisifier([1, 1], group, lables, 3))

#test file2matrix
group, lables = file2matrix('/Users/kuangwentao/PycharmProjects/ML_Algorithm/kNN/DataSets/datingTestSet.txt', 3)
print(group)
print(lables)
