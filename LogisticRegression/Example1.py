'''
Simple two dimension classification example for testing the algorithm

@author: Wentao Kuang
'''

from logReg import *
import matplotlib.pyplot as plt
from numpy import *

'''
load dataset
'''
def loadDataSet():
    dataMat=[]
    labelMat=[]
    f=open('/Users/kuangwentao/PycharmProjects/ML_Algorithm/DataSets/logReg/testSet.txt')
    for line in f.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


dataMat, labelMat =loadDataSet();
weights = gradAscent(dataMat,labelMat,0.001,500)
print weights
'''
plot result
'''
def plotBestFit():
    dataArr = array(dataMat)
    n = shape(dataMat)[0]
    xcord1=[]
    xcord2=[]
    ycord1=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y.transpose())
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.show()

plotBestFit()

