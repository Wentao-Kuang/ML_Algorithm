import matplotlib
import matplotlib.pyplot as plt
import kNN as kNN
import numpy as np

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

