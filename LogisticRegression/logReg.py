'''
Logistic Regression Algorithm

@author: Wentao Kuang
'''

from numpy import *

'''
sigmoid function
'''
def sigmoid(x):
    return 1.0/(1+exp(-x))

'''
Gradient Ascent based optimization

Input:      dataMat: dataset features matrix
            labelMat: labels matrix
            alpha: moving rate
            cycles: moving iterations

Output:     trained weights
      
'''
def gradAscent(dataMat,labelMat,alpha,cycles):
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).transpose()
    m,n=shape(dataMatrix)
    weights=ones((n,1))
    for k in range(cycles):
        h = sigmoid(dataMatrix*weights) #hypothesis
        error = (labelMatrix-h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

