import numpy as np
from kNN import *
'''
kNN: Testing k Nearest Neighbors

@author: Wentao Kuang
'''

#create test dataset
def createTestDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

#test kNN with examle dataset
group, lables=createTestDataSet()
print(kNNClisifier([1, 1], group, lables, 3))


#test datingTestSet
group, lables = file2matrix('/Users/kuangwentao/PycharmProjects/ML_Algorithm/kNN/DataSets/datingTestSet.txt', 3)
print(group)
print(lables)

#test featurre normalization
group, lables=createTestDataSet()
normgroup, ranges, minVal = norm(group)
print(normgroup)
print(ranges)
print(minVal)