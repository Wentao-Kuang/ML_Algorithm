import numpy as np
from Tree import *
from TreePlot import *
'''
kNN: Testing Decision Tree

@author: Wentao Kuang
'''

#Create Test data Set
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

group, labels = createDataSet()

#print splitDataSet(group,0,1)
#print(chooseBestFeature(group))
tree=createTree(group, labels)
labels = ['no surfacing', 'flippers']
print(classify(tree,labels,[1,1]))
storeTree(tree,'Example1Tree.txt')
print(grabTree('Example1Tree.txt'))
#createPlot(tree)