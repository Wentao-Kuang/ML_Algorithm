from Tree import *
from TreePlot import *
'''
Lenses data traing
Using lenses dataset

@author: Wentao Kuang
'''

'''
read data
'''
def readLense(file):
    fr=open(file)
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lensesLables=['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses,lensesLables

lenses,lensesLables=readLense('/Users/kuangwentao/PycharmProjects/ML_Algorithm/DataSets/lenses.txt')
lensesTree=createTree(lenses,lensesLables)
createPlot(lensesTree)