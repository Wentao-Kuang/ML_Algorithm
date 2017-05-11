import kNN as kNN
import numpy as np
from os import listdir
import operator

'''
Image number recognition

@author: Wentao Kuang
'''


'''
Read image file and convert to vector

Input:      filename: file address

Output:     returnMat: converted observations matrix

'''
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


'''
Dataset construction

Input:      filedirectory: datasets folder

Output:     group, lables
'''
def dataConstruction(filedirectory):
    fileList = listdir(filedirectory)
    m = len(fileList)
    lables = []
    returnMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = fileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        lables.append(classNumStr)
        returnMat[i, :] = img2vector(filedirectory+"/"+fileNameStr)
    return returnMat,lables


'''
Testing the hand writing number recognition
'''
def NumRecTesting():
    # training data construction
    trainingMat, lables = dataConstruction('/Users/kuangwentao/PycharmProjects/ML_Algorithm/kNN/DataSets/trainingDigits')

    # test data construction and testing error rate
    testFileList = listdir('/Users/kuangwentao/PycharmProjects/ML_Algorithm/kNN/DataSets/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/Users/kuangwentao/PycharmProjects/ML_Algorithm/kNN/DataSets/testDigits/%s' % fileNameStr)
        result = kNN.kNNClisifier(vectorUnderTest, trainingMat, lables, 3)
        if(result != classNumStr) : errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

NumRecTesting()
