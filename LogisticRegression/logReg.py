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


'''
Stochastic Gradient Ascent based optimization

Input:      dataMat: dataset features matrix
            labelMat: labels matrix
            alpha: moving rate

Output:     trained weights

'''
def stocGradAscent(dataMat,labelMat,alpha,cycles):
    m,n=shape(dataMat)
    weights=ones(n)
    for j in range(cycles):
        dataIndex=range(m)
        for i in range(m):
            a = 4/(1.0+j+i)+alpha # convergent moving rate
            randomIndex = int(random.uniform(0,len(dataIndex)))#random pick update
            h = sigmoid(sum(dataMat[randomIndex]*weights))
            error = labelMat[randomIndex] - h
            weights = weights + a*error*dataMat[randomIndex]
            del(dataIndex[randomIndex])
    return weights